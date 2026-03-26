use std::sync::RwLock;

use rustpython_vm as vm;
use vm::atomic_func;
use vm::builtins::{PyDict, PyList, PyStr};
use vm::protocol::{PyMappingMethods, PySequenceMethods};
use vm::types::{AsMapping, AsSequence, Representable};
use vm::{Py, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};

use pandas_rust_core::column::{Column, ColumnData};
use pandas_rust_core::ops::{aggregation, filter, math, nulls, sort};
use pandas_rust_core::{DType, DataFrame};

use crate::py_column::{
    agg_result_to_pyobj, column_to_pylist, pandas_err, pyobj_to_column, pyobj_to_scalar_value,
};
use crate::py_groupby::PyGroupBy;
use crate::py_series::PySeries;

/// Python-visible DataFrame class wrapping pandas_rust_core::DataFrame.
/// Uses `RwLock` for interior mutability so `__setitem__` can work
/// through RustPython's `&self` method signatures.
#[vm::pyclass(module = "_pandas_native", name = "DataFrame")]
#[derive(Debug, PyPayload)]
pub struct PyDataFrame {
    pub(crate) data: RwLock<DataFrame>,
}

impl Clone for PyDataFrame {
    fn clone(&self) -> Self {
        Self {
            data: RwLock::new(self.data.read().unwrap().clone()),
        }
    }
}

impl PyDataFrame {
    pub fn from_core(df: DataFrame) -> Self {
        Self {
            data: RwLock::new(df),
        }
    }

    pub fn to_py(self, vm: &VirtualMachine) -> PyObjectRef {
        self.into_pyobject(vm)
    }

    /// Convenience: read-lock the inner DataFrame.
    fn inner(&self) -> std::sync::RwLockReadGuard<'_, DataFrame> {
        self.data.read().unwrap()
    }
}

#[vm::pyclass(with(Representable, AsMapping, AsSequence))]
impl PyDataFrame {
    /// DataFrame(data_dict) constructor.
    #[pyslot]
    fn slot_new(
        cls: vm::builtins::PyTypeRef,
        args: vm::function::FuncArgs,
        vm: &VirtualMachine,
    ) -> PyResult {
        // Accept a single positional argument: a dict
        let data_obj = args
            .args
            .first()
            .cloned()
            .unwrap_or_else(|| vm.ctx.new_dict().into());

        let dict = data_obj
            .downcast_ref::<PyDict>()
            .ok_or_else(|| vm.new_type_error("DataFrame() requires a dict".to_owned()))?;

        let mut columns = Vec::new();
        for (key, value) in dict.into_iter() {
            let name = key
                .downcast_ref::<PyStr>()
                .ok_or_else(|| vm.new_type_error("dict keys must be strings".to_owned()))?;
            let col = pyobj_to_column(name.as_str(), &value, vm)?;
            columns.push(col);
        }

        let df = DataFrame::from_columns(columns).map_err(|e| pandas_err(e, vm))?;

        Ok(PyDataFrame::from_core(df)
            .into_ref_with_type(vm, cls)?
            .into())
    }

    // --- Properties ---

    #[pygetset]
    fn shape(&self, vm: &VirtualMachine) -> PyObjectRef {
        let (r, c) = self.inner().shape();
        vm.ctx
            .new_tuple(vec![vm.ctx.new_int(r).into(), vm.ctx.new_int(c).into()])
            .into()
    }

    #[pygetset]
    fn dtypes(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let dict = vm.ctx.new_dict();
        for (name, dtype) in self.inner().dtypes() {
            dict.set_item(name, vm.ctx.new_str(dtype.to_string()).into(), vm)?;
        }
        Ok(dict.into())
    }

    #[pygetset]
    fn columns(&self, vm: &VirtualMachine) -> PyObjectRef {
        let items: Vec<PyObjectRef> = self
            .inner()
            .column_names()
            .iter()
            .map(|n| vm.ctx.new_str(*n).into())
            .collect();
        vm.ctx.new_list(items).into()
    }

    #[pymethod]
    fn __len__(&self) -> usize {
        self.inner().nrows()
    }

    // --- Indexing ---

    #[pymethod]
    fn __getitem__(&self, key: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // String key -> PySeries
        if let Some(s) = key.downcast_ref::<PyStr>() {
            let series = self
                .inner()
                .get_series(s.as_str())
                .map_err(|e| pandas_err(e, vm))?;
            return Ok(PySeries::from_core(series).into_pyobject(vm));
        }

        // List of strings -> PyDataFrame with selected columns
        if let Some(list) = key.downcast_ref::<PyList>() {
            let items = list.borrow_vec();
            let names: PyResult<Vec<String>> = items
                .iter()
                .map(|item| {
                    item.downcast_ref::<PyStr>()
                        .map(|s| s.as_str().to_owned())
                        .ok_or_else(|| vm.new_type_error("column names must be strings".to_owned()))
                })
                .collect();
            let names = names?;
            let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            let sub = self
                .inner()
                .select_columns(&name_refs)
                .map_err(|e| pandas_err(e, vm))?;
            return Ok(PyDataFrame::from_core(sub).into_pyobject(vm));
        }

        // PySeries (Bool) -> filter rows
        if let Some(mask_series) = key.downcast_ref::<PySeries>() {
            let mask_col = mask_series.inner.column();
            let indices = filter::filter_indices(mask_col).map_err(|e| pandas_err(e, vm))?;
            let sub = self
                .inner()
                .take_rows(&indices)
                .map_err(|e| pandas_err(e, vm))?;
            return Ok(PyDataFrame::from_core(sub).into_pyobject(vm));
        }

        Err(vm.new_type_error("unsupported key type for DataFrame.__getitem__".to_owned()))
    }

    #[pymethod]
    fn __setitem__(
        &self,
        key: PyObjectRef,
        value: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        let name = key
            .downcast_ref::<PyStr>()
            .ok_or_else(|| vm.new_type_error("key must be a string".to_owned()))?;

        let col = pyobj_to_column(name.as_str(), &value, vm)?;

        let mut df = self.data.write().unwrap();
        df.set_column(col).map_err(|e| pandas_err(e, vm))?;
        Ok(())
    }

    // --- Column access ---

    #[pymethod]
    fn get_column(&self, name: PyObjectRef, vm: &VirtualMachine) -> PyResult<PySeries> {
        let s = name
            .downcast_ref::<PyStr>()
            .ok_or_else(|| vm.new_type_error("column name must be a string".to_owned()))?;
        let series = self
            .inner()
            .get_series(s.as_str())
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PySeries::from_core(series))
    }

    // --- head / tail ---

    #[pymethod]
    fn head(
        &self,
        n: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyDataFrame> {
        let n = n.unwrap_or(5).min(self.inner().nrows());
        let indices: Vec<usize> = (0..n).collect();
        let sub = self
            .inner()
            .take_rows(&indices)
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(sub))
    }

    #[pymethod]
    fn tail(
        &self,
        n: vm::function::OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyDataFrame> {
        let n = n.unwrap_or(5).min(self.inner().nrows());
        let start = self.inner().nrows().saturating_sub(n);
        let indices: Vec<usize> = (start..self.inner().nrows()).collect();
        let sub = self
            .inner()
            .take_rows(&indices)
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(sub))
    }

    // --- sort ---

    #[pymethod]
    fn sort_values(
        &self,
        by: PyObjectRef,
        ascending: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyDataFrame> {
        let asc = ascending.unwrap_or(true);
        let col_name = by
            .downcast_ref::<PyStr>()
            .ok_or_else(|| vm.new_type_error("'by' must be a string".to_owned()))?;
        let df = self.inner();
        let col = df
            .get_column(col_name.as_str())
            .map_err(|e| pandas_err(e, vm))?;
        let indices = sort::argsort_column(col, asc);
        let sub = df.take_rows(&indices).map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(sub))
    }

    // --- drop ---

    #[pymethod]
    fn drop(&self, columns: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        let to_drop: Vec<String> = if let Some(s) = columns.downcast_ref::<PyStr>() {
            vec![s.as_str().to_owned()]
        } else if let Some(list) = columns.downcast_ref::<PyList>() {
            let items = list.borrow_vec();
            items
                .iter()
                .map(|item| {
                    item.downcast_ref::<PyStr>()
                        .map(|s| s.as_str().to_owned())
                        .ok_or_else(|| vm.new_type_error("column names must be strings".to_owned()))
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            return Err(vm.new_type_error("columns must be a string or list of strings".to_owned()));
        };

        let drop_set: std::collections::HashSet<&str> =
            to_drop.iter().map(|s| s.as_str()).collect();
        let df = self.inner();
        let keep: Vec<&str> = df
            .column_names()
            .into_iter()
            .filter(|n| !drop_set.contains(n))
            .collect();
        let sub = df.select_columns(&keep).map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(sub))
    }

    // --- rename ---

    #[pymethod]
    fn rename(&self, columns: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        let dict = columns
            .downcast_ref::<PyDict>()
            .ok_or_else(|| vm.new_type_error("columns must be a dict".to_owned()))?;

        let mut mapping = Vec::new();
        for (k, v) in dict.into_iter() {
            let old = k
                .downcast_ref::<PyStr>()
                .ok_or_else(|| vm.new_type_error("keys must be strings".to_owned()))?
                .as_str()
                .to_owned();
            let new = v
                .downcast_ref::<PyStr>()
                .ok_or_else(|| vm.new_type_error("values must be strings".to_owned()))?
                .as_str()
                .to_owned();
            mapping.push((old, new));
        }
        let mapping_refs: Vec<(&str, &str)> = mapping
            .iter()
            .map(|(o, n)| (o.as_str(), n.as_str()))
            .collect();
        let df = self
            .inner()
            .rename_columns(&mapping_refs)
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(df))
    }

    // --- duplicated / drop_duplicates ---

    #[pymethod]
    fn duplicated(
        &self,
        subset: vm::function::OptionalArg<PyObjectRef>,
        keep: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PySeries> {
        use pandas_rust_core::Series;
        let keep_enum = crate::py_series::parse_keep_arg(keep.into_option().as_ref(), vm)?;
        let df = self.inner();
        let subset_names = parse_optional_subset(subset.into_option().as_ref(), vm)?;
        let subset_refs: Option<Vec<&str>> = subset_names
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let col = df
            .duplicated_rows(subset_refs.as_deref(), keep_enum)
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PySeries::from_core(Series::new(col)))
    }

    #[pymethod]
    fn drop_duplicates(
        &self,
        subset: vm::function::OptionalArg<PyObjectRef>,
        keep: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyDataFrame> {
        let keep_enum = crate::py_series::parse_keep_arg(keep.into_option().as_ref(), vm)?;
        let df = self.inner();
        let subset_names = parse_optional_subset(subset.into_option().as_ref(), vm)?;
        let subset_refs: Option<Vec<&str>> = subset_names
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let result = df
            .drop_duplicates(subset_refs.as_deref(), keep_enum)
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(result))
    }

    // --- copy ---

    #[pymethod]
    fn copy(&self) -> PyDataFrame {
        self.clone()
    }

    // --- Aggregations ---

    #[pymethod]
    fn sum(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.agg_all(aggregation::sum, vm)
    }

    #[pymethod]
    fn mean(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.agg_all(aggregation::mean, vm)
    }

    #[pymethod]
    fn min(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.agg_all(aggregation::min, vm)
    }

    #[pymethod]
    fn max(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.agg_all(aggregation::max, vm)
    }

    #[pymethod]
    fn count(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let dict = vm.ctx.new_dict();
        for (name, col) in self.inner().iter_columns() {
            let r = aggregation::count(col);
            dict.set_item(name, agg_result_to_pyobj(&r, vm), vm)?;
        }
        Ok(dict.into())
    }

    #[pymethod]
    fn std(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.agg_all_with_ddof(|c| aggregation::std(c, 1), vm)
    }

    #[pymethod]
    fn var(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.agg_all_with_ddof(|c| aggregation::var(c, 1), vm)
    }

    #[pymethod]
    fn median(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.agg_all(aggregation::median, vm)
    }

    // --- Null operations ---

    #[pymethod]
    fn fillna(&self, value: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        let sv = pyobj_to_scalar_value(&value, vm)?;
        let mut cols = Vec::new();
        for (_, col) in self.inner().iter_columns() {
            let new_col = nulls::fillna_scalar(col, &sv).map_err(|e| pandas_err(e, vm))?;
            cols.push(new_col);
        }
        let df = DataFrame::from_columns(cols).map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(df))
    }

    #[pymethod]
    fn dropna(
        &self,
        how: vm::function::OptionalArg<PyRef<PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyDataFrame> {
        let how_str = how.as_ref().map(|s| s.as_str()).unwrap_or("any");
        let drop_how = match how_str {
            "any" => nulls::DropHow::Any,
            "all" => nulls::DropHow::All,
            _ => {
                return Err(
                    vm.new_value_error(format!("how must be 'any' or 'all', got '{}'", how_str))
                )
            }
        };
        let df = self.inner();
        let col_refs: Vec<&Column> = df.iter_columns().map(|(_, c)| c).collect();
        let keep_indices = nulls::dropna_rows(&col_refs, drop_how);
        let sub = df.take_rows(&keep_indices).map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(sub))
    }

    #[pymethod]
    fn isna(&self) -> PyDataFrame {
        let cols: Vec<Column> = self
            .inner()
            .iter_columns()
            .map(|(_, col)| nulls::isna(col))
            .collect();
        let df = DataFrame::from_columns(cols).expect("isna columns have same length");
        PyDataFrame::from_core(df)
    }

    #[pymethod]
    fn notna(&self) -> PyDataFrame {
        let cols: Vec<Column> = self
            .inner()
            .iter_columns()
            .map(|(_, col)| nulls::notna(col))
            .collect();
        let df = DataFrame::from_columns(cols).expect("notna columns have same length");
        PyDataFrame::from_core(df)
    }

    // --- GroupBy ---

    #[pymethod]
    fn groupby(&self, by: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyGroupBy> {
        let by_cols: Vec<String> = if let Some(s) = by.downcast_ref::<PyStr>() {
            vec![s.as_str().to_owned()]
        } else if let Some(list) = by.downcast_ref::<PyList>() {
            let items = list.borrow_vec();
            items
                .iter()
                .map(|item| {
                    item.downcast_ref::<PyStr>()
                        .map(|s| s.as_str().to_owned())
                        .ok_or_else(|| vm.new_type_error("groupby keys must be strings".to_owned()))
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            return Err(vm.new_type_error("by must be a string or list of strings".to_owned()));
        };

        Ok(PyGroupBy::new(self.inner().clone(), by_cols))
    }

    // --- Merge ---

    #[pymethod]
    fn merge(&self, args: vm::function::FuncArgs, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        // merge(right, on=..., how="inner", suffixes=("_x","_y"))
        let right_obj = args
            .args
            .first()
            .ok_or_else(|| vm.new_type_error("merge() requires a right DataFrame".to_owned()))?;
        let right_df = right_obj
            .downcast_ref::<PyDataFrame>()
            .ok_or_else(|| vm.new_type_error("right must be a DataFrame".to_owned()))?;

        // on
        let on_obj = args.kwargs.get("on").cloned().ok_or_else(|| {
            vm.new_type_error("merge() requires 'on' keyword argument".to_owned())
        })?;
        let on_cols: Vec<String> = if let Some(s) = on_obj.downcast_ref::<PyStr>() {
            vec![s.as_str().to_owned()]
        } else if let Some(list) = on_obj.downcast_ref::<PyList>() {
            let items = list.borrow_vec();
            items
                .iter()
                .map(|item| {
                    item.downcast_ref::<PyStr>()
                        .map(|s| s.as_str().to_owned())
                        .ok_or_else(|| vm.new_type_error("on keys must be strings".to_owned()))
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            return Err(vm.new_type_error("on must be a string or list of strings".to_owned()));
        };

        // how
        let how_str = args
            .kwargs
            .get("how")
            .and_then(|o| o.downcast_ref::<PyStr>().map(|s| s.as_str().to_owned()))
            .unwrap_or_else(|| "inner".to_owned());
        let how = match how_str.as_str() {
            "inner" => pandas_rust_core::merge::JoinHow::Inner,
            "left" => pandas_rust_core::merge::JoinHow::Left,
            "right" => pandas_rust_core::merge::JoinHow::Right,
            "outer" => pandas_rust_core::merge::JoinHow::Outer,
            _ => {
                return Err(vm.new_value_error(format!("unknown join type: {}", how_str)));
            }
        };

        // suffixes
        let (suffix_l, suffix_r) = if let Some(suf_obj) = args.kwargs.get("suffixes") {
            if let Some(tup) = suf_obj.downcast_ref::<vm::builtins::PyTuple>() {
                let elems = tup.as_slice();
                if elems.len() != 2 {
                    return Err(
                        vm.new_value_error("suffixes must be a tuple of two strings".to_owned())
                    );
                }
                let l = elems[0]
                    .downcast_ref::<PyStr>()
                    .ok_or_else(|| vm.new_type_error("suffix must be a string".to_owned()))?
                    .as_str()
                    .to_owned();
                let r = elems[1]
                    .downcast_ref::<PyStr>()
                    .ok_or_else(|| vm.new_type_error("suffix must be a string".to_owned()))?
                    .as_str()
                    .to_owned();
                (l, r)
            } else {
                return Err(vm.new_type_error("suffixes must be a tuple".to_owned()));
            }
        } else {
            ("_x".to_owned(), "_y".to_owned())
        };

        let on_refs: Vec<&str> = on_cols.iter().map(|s| s.as_str()).collect();
        let self_df = self.inner();
        let right_inner = right_df.data.read().unwrap();
        let result = pandas_rust_core::merge::merge(
            &self_df,
            &right_inner,
            &on_refs,
            how,
            (&suffix_l, &suffix_r),
        )
        .map_err(|e| pandas_err(e, vm))?;

        Ok(PyDataFrame::from_core(result))
    }

    // --- I/O ---

    #[pymethod]
    fn to_csv(
        &self,
        path: vm::function::OptionalArg<PyRef<PyStr>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let options = pandas_rust_core::io::csv::CsvWriteOptions::default();
        let df = self.inner();
        match path.into_option() {
            Some(p) => {
                let file = std::fs::File::create(p.as_str())
                    .map_err(|e| vm.new_value_error(e.to_string()))?;
                pandas_rust_core::io::csv::to_csv(&df, file, options)
                    .map_err(|e| pandas_err(e, vm))?;
                Ok(vm.ctx.none())
            }
            None => {
                let mut buf = Vec::new();
                pandas_rust_core::io::csv::to_csv(&df, &mut buf, options)
                    .map_err(|e| pandas_err(e, vm))?;
                let s = String::from_utf8(buf).map_err(|e| vm.new_value_error(e.to_string()))?;
                Ok(vm.ctx.new_str(s).into())
            }
        }
    }

    // --- to_dict ---

    #[pymethod]
    fn to_dict(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let dict = vm.ctx.new_dict();
        for (name, col) in self.inner().iter_columns() {
            let list = column_to_pylist(col, vm)?;
            dict.set_item(name, list, vm)?;
        }
        Ok(dict.into())
    }

    // --- describe ---

    #[pymethod]
    fn describe(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        // Only numeric columns
        let mut result_cols = Vec::new();
        for (name, col) in self.inner().iter_columns() {
            let dt = col.dtype();
            if dt != DType::Float64 && dt != DType::Int64 {
                continue;
            }
            let count_r = aggregation::count(col);
            let mean_r = aggregation::mean(col).unwrap_or(aggregation::AggResult::None);
            let std_r = aggregation::std(col, 1).unwrap_or(aggregation::AggResult::None);
            let min_r = aggregation::min(col).unwrap_or(aggregation::AggResult::None);
            let max_r = aggregation::max(col).unwrap_or(aggregation::AggResult::None);

            let to_f64 = |r: &aggregation::AggResult| -> f64 {
                match r {
                    aggregation::AggResult::Int64(v) => *v as f64,
                    aggregation::AggResult::Float64(v) => *v,
                    aggregation::AggResult::Usize(v) => *v as f64,
                    _ => f64::NAN,
                }
            };

            let vals = vec![
                to_f64(&count_r),
                to_f64(&mean_r),
                to_f64(&std_r),
                to_f64(&min_r),
                to_f64(&max_r),
            ];
            result_cols.push(Column::new(name, ColumnData::Float64(vals)));
        }

        if result_cols.is_empty() {
            return Ok(PyDataFrame::from_core(DataFrame::empty()));
        }

        let df = DataFrame::from_columns(result_cols).map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(df))
    }

    // --- numpy interop ---

    #[pymethod]
    fn to_numpy(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        use numpy_rust_core::NdArray;
        use numpy_rust_python::py_array::PyNdArray;

        let df = self.inner();
        let nrows = df.nrows();
        let ncols = df.ncols();

        if ncols == 0 || nrows == 0 {
            let arr = NdArray::from_vec(Vec::<f64>::new());
            return Ok(PyNdArray::from_core(arr).into_pyobject(vm));
        }

        // Check if all columns are numeric
        let all_numeric = df
            .iter_columns()
            .all(|(_, col)| matches!(col.dtype(), DType::Int64 | DType::Float64 | DType::Bool));

        if !all_numeric {
            return Err(vm.new_type_error(
                "Cannot convert DataFrame with string columns to numpy array. Use select_dtypes first.".to_owned(),
            ));
        }

        // Build row-major flat vector (row 0 col 0, row 0 col 1, ...)
        let mut flat = Vec::with_capacity(nrows * ncols);
        for row_idx in 0..nrows {
            for (_, col) in df.iter_columns() {
                if col.is_null(row_idx) {
                    flat.push(f64::NAN);
                } else {
                    match col.data() {
                        ColumnData::Bool(v) => flat.push(if v[row_idx] { 1.0 } else { 0.0 }),
                        ColumnData::Int64(v) => flat.push(v[row_idx] as f64),
                        ColumnData::Float64(v) => flat.push(v[row_idx]),
                        ColumnData::Str(_) => unreachable!(),
                    }
                }
            }
        }

        let arr = NdArray::from_vec(flat)
            .reshape(&[nrows, ncols])
            .map_err(|e| vm.new_value_error(e.to_string()))?;
        Ok(PyNdArray::from_core(arr).into_pyobject(vm))
    }

    #[pygetset]
    fn values(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.to_numpy(vm)
    }

    // --- abs ---

    #[pymethod]
    fn abs(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        let mut cols = Vec::new();
        for (_, col) in self.inner().iter_columns() {
            let new_col = match col.dtype() {
                DType::Int64 | DType::Float64 => {
                    math::abs_column(col).map_err(|e| pandas_err(e, vm))?
                }
                _ => col.clone(),
            };
            cols.push(new_col);
        }
        let df = DataFrame::from_columns(cols).map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(df))
    }

    // --- clip ---

    #[pymethod]
    fn clip(
        &self,
        lower: vm::function::OptionalArg<PyObjectRef>,
        upper: vm::function::OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyDataFrame> {
        let lo = lower
            .into_option()
            .filter(|obj| !vm.is_none(obj))
            .map(|obj| -> PyResult<f64> { obj.try_into_value(vm) })
            .transpose()?;
        let hi = upper
            .into_option()
            .filter(|obj| !vm.is_none(obj))
            .map(|obj| -> PyResult<f64> { obj.try_into_value(vm) })
            .transpose()?;
        let mut cols = Vec::new();
        for (_, col) in self.inner().iter_columns() {
            let new_col = match col.dtype() {
                DType::Int64 | DType::Float64 => {
                    math::clip_column(col, lo, hi).map_err(|e| pandas_err(e, vm))?
                }
                _ => col.clone(),
            };
            cols.push(new_col);
        }
        let df = DataFrame::from_columns(cols).map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(df))
    }

    // --- nlargest ---

    #[pymethod]
    fn nlargest(
        &self,
        n: usize,
        columns: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyDataFrame> {
        let col_name = columns
            .downcast_ref::<PyStr>()
            .ok_or_else(|| vm.new_type_error("columns must be a string".to_owned()))?;
        let df = self.inner();
        let col = df
            .get_column(col_name.as_str())
            .map_err(|e| pandas_err(e, vm))?;
        let indices = sort::argsort_column(col, false);
        let take_n = n.min(indices.len());
        let sub = df
            .take_rows(&indices[..take_n])
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(sub))
    }

    // --- nsmallest ---

    #[pymethod]
    fn nsmallest(
        &self,
        n: usize,
        columns: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyDataFrame> {
        let col_name = columns
            .downcast_ref::<PyStr>()
            .ok_or_else(|| vm.new_type_error("columns must be a string".to_owned()))?;
        let df = self.inner();
        let col = df
            .get_column(col_name.as_str())
            .map_err(|e| pandas_err(e, vm))?;
        let indices = sort::argsort_column(col, true);
        let take_n = n.min(indices.len());
        let sub = df
            .take_rows(&indices[..take_n])
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(sub))
    }

    // --- transpose ---

    #[pymethod]
    fn transpose(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        let result = self.inner().transpose().map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(result))
    }

    // --- iterrows ---

    #[pymethod]
    fn iterrows(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let df = self.inner();
        let col_names: Vec<String> = df.column_names().iter().map(|s| s.to_string()).collect();
        let nrows = df.nrows();

        let mut rows = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let dict = vm.ctx.new_dict();
            for name in &col_names {
                let col = df.get_column(name).unwrap();
                let val = crate::py_column::column_value_to_pyobj(col, i, vm)?;
                dict.set_item(name.as_str(), val, vm)?;
            }
            let idx_val: PyObjectRef = vm.ctx.new_int(i).into();
            let tuple = vm.ctx.new_tuple(vec![idx_val, dict.into()]);
            rows.push(tuple.into());
        }
        Ok(vm.ctx.new_list(rows).into())
    }

    // --- itertuples ---

    #[pymethod]
    fn itertuples(
        &self,
        index: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let include_index = index.unwrap_or(true);
        let df = self.inner();
        let nrows = df.nrows();
        let col_names: Vec<String> = df.column_names().iter().map(|s| s.to_string()).collect();

        let mut rows = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let mut vals = Vec::new();
            if include_index {
                vals.push(vm.ctx.new_int(i).into());
            }
            for name in &col_names {
                let col = df.get_column(name).unwrap();
                vals.push(crate::py_column::column_value_to_pyobj(col, i, vm)?);
            }
            rows.push(vm.ctx.new_tuple(vals).into());
        }
        Ok(vm.ctx.new_list(rows).into())
    }

    // --- apply ---

    #[pymethod]
    fn apply(
        &self,
        func: PyObjectRef,
        axis: vm::function::OptionalArg<i32>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let axis = axis.unwrap_or(0);
        let df = self.inner();

        if axis == 0 {
            // Per-column: call func(series) for each column
            let dict = vm.ctx.new_dict();
            for (name, col) in df.iter_columns() {
                let series = crate::py_series::PySeries::from_core(pandas_rust_core::Series::new(
                    col.clone(),
                ));
                let series_obj: PyObjectRef = series.into_pyobject(vm);
                let result = func.call((series_obj,), vm)?;
                dict.set_item(name, result, vm)?;
            }
            Ok(dict.into())
        } else {
            // Per-row: call func(row_dict) for each row
            let col_names: Vec<String> = df.column_names().iter().map(|s| s.to_string()).collect();
            let nrows = df.nrows();
            let mut results = Vec::with_capacity(nrows);

            for i in 0..nrows {
                let dict = vm.ctx.new_dict();
                for name in &col_names {
                    let col = df.get_column(name).unwrap();
                    let val = crate::py_column::column_value_to_pyobj(col, i, vm)?;
                    dict.set_item(name.as_str(), val, vm)?;
                }
                let dict_obj: PyObjectRef = dict.into();
                let result = func.call((dict_obj,), vm)?;
                results.push(result);
            }

            Ok(vm.ctx.new_list(results).into())
        }
    }

    // --- applymap ---

    #[pymethod]
    fn applymap(&self, func: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        let df = self.inner();
        let mut new_cols = Vec::new();

        for (name, col) in df.iter_columns() {
            let mut new_values = Vec::with_capacity(col.len());
            for i in 0..col.len() {
                let val = crate::py_column::column_value_to_pyobj(col, i, vm)?;
                let result = func.call((val,), vm)?;
                new_values.push(result);
            }
            let list_obj = vm.ctx.new_list(new_values);
            let new_col = crate::py_column::pyobj_to_column(name, &list_obj.into(), vm)?;
            new_cols.push(new_col);
        }

        let result =
            pandas_rust_core::DataFrame::from_columns(new_cols).map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(result))
    }
}

// --- Module-level helpers ---

fn parse_optional_subset(
    subset: Option<&PyObjectRef>,
    vm: &VirtualMachine,
) -> PyResult<Option<Vec<String>>> {
    match subset {
        None => Ok(None),
        Some(obj) => {
            if vm.is_none(obj) {
                return Ok(None);
            }
            if let Some(s) = obj.downcast_ref::<PyStr>() {
                return Ok(Some(vec![s.as_str().to_owned()]));
            }
            if let Some(list) = obj.downcast_ref::<PyList>() {
                let items = list.borrow_vec();
                let names: PyResult<Vec<String>> = items
                    .iter()
                    .map(|item| {
                        item.downcast_ref::<PyStr>()
                            .map(|s| s.as_str().to_owned())
                            .ok_or_else(|| {
                                vm.new_type_error("subset names must be strings".to_owned())
                            })
                    })
                    .collect();
                return Ok(Some(names?));
            }
            Err(vm.new_type_error("subset must be a string or list of strings".to_owned()))
        }
    }
}

// --- Private helper methods ---
impl PyDataFrame {
    fn agg_all(
        &self,
        f: fn(&Column) -> pandas_rust_core::Result<aggregation::AggResult>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let dict = vm.ctx.new_dict();
        for (name, col) in self.inner().iter_columns() {
            match f(col) {
                Ok(r) => {
                    dict.set_item(name, agg_result_to_pyobj(&r, vm), vm)?;
                }
                Err(_) => {
                    // Skip non-numeric columns for aggregation
                }
            }
        }
        Ok(dict.into())
    }

    fn agg_all_with_ddof(
        &self,
        f: fn(&Column) -> pandas_rust_core::Result<aggregation::AggResult>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        self.agg_all(f, vm)
    }
}

impl AsSequence for PyDataFrame {
    fn as_sequence() -> &'static PySequenceMethods {
        use std::sync::LazyLock as Lazy;
        static AS_SEQUENCE: Lazy<PySequenceMethods> = Lazy::new(|| PySequenceMethods {
            length: atomic_func!(|seq, _vm| {
                let zelf = PyDataFrame::sequence_downcast(seq);
                Ok(zelf.inner().nrows())
            }),
            ..PySequenceMethods::NOT_IMPLEMENTED
        });
        &AS_SEQUENCE
    }
}

impl AsMapping for PyDataFrame {
    fn as_mapping() -> &'static PyMappingMethods {
        use std::sync::LazyLock as Lazy;
        static AS_MAPPING: Lazy<PyMappingMethods> =
            Lazy::new(|| PyMappingMethods {
                length: atomic_func!(|mapping, _vm| {
                    let zelf = PyDataFrame::mapping_downcast(mapping);
                    Ok(zelf.inner().nrows())
                }),
                subscript: atomic_func!(|mapping, needle: &vm::PyObject, vm| {
                    let zelf = PyDataFrame::mapping_downcast(mapping);
                    zelf.__getitem__(needle.to_owned(), vm)
                }),
                ass_subscript: atomic_func!(|mapping, needle: &vm::PyObject, value, vm| {
                    let zelf = PyDataFrame::mapping_downcast(mapping);
                    match value {
                        Some(value) => {
                            zelf.__setitem__(needle.to_owned(), value.to_owned(), vm)?;
                            Ok(())
                        }
                        None => Err(vm
                            .new_type_error("DataFrame does not support item deletion".to_owned())),
                    }
                }),
            });
        &AS_MAPPING
    }
}

impl Representable for PyDataFrame {
    fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
        let df = zelf.data.read().unwrap();
        let nrows = df.nrows();
        let ncols = df.ncols();

        if ncols == 0 {
            return Ok("Empty DataFrame".to_string());
        }

        let col_names = df.column_names();
        let max_rows = 10;
        let show_rows = nrows.min(max_rows);

        // Compute column widths
        let mut col_widths: Vec<usize> = col_names.iter().map(|n| n.len()).collect();

        // Gather string representations
        let mut cell_strs: Vec<Vec<String>> = Vec::new();
        for (ci, &cname) in col_names.iter().enumerate() {
            let col = df.get_column(cname).unwrap();
            let mut cells = Vec::new();
            for ri in 0..show_rows {
                let s = if col.is_null(ri) {
                    "NaN".to_string()
                } else {
                    match col.data() {
                        ColumnData::Bool(v) => format!("{}", v[ri]),
                        ColumnData::Int64(v) => format!("{}", v[ri]),
                        ColumnData::Float64(v) => format!("{}", v[ri]),
                        ColumnData::Str(v) => v[ri].clone(),
                    }
                };
                if s.len() > col_widths[ci] {
                    col_widths[ci] = s.len();
                }
                cells.push(s);
            }
            cell_strs.push(cells);
        }

        // Index width
        let idx_width = if nrows > 0 {
            format!("{}", show_rows - 1).len()
        } else {
            1
        };

        let mut out = String::new();

        // Header
        out.push_str(&" ".repeat(idx_width + 2));
        for (ci, &cname) in col_names.iter().enumerate() {
            out.push_str(&format!("{:>width$}", cname, width = col_widths[ci]));
            if ci < ncols - 1 {
                out.push_str("  ");
            }
        }
        out.push('\n');

        // Rows
        #[allow(clippy::needless_range_loop)]
        for ri in 0..show_rows {
            out.push_str(&format!("{:<width$}  ", ri, width = idx_width));
            for (ci, _) in col_names.iter().enumerate() {
                out.push_str(&format!(
                    "{:>width$}",
                    cell_strs[ci][ri],
                    width = col_widths[ci]
                ));
                if ci < ncols - 1 {
                    out.push_str("  ");
                }
            }
            out.push('\n');
        }

        if nrows > max_rows {
            out.push_str(&format!("... ({} rows total)\n", nrows));
        }

        out.push_str(&format!("\n[{} rows x {} columns]", nrows, ncols));
        Ok(out)
    }
}

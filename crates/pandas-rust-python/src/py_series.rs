use rustpython_vm as vm;
use vm::builtins::PyStr;
use vm::protocol::PyNumberMethods;
use vm::types::{AsNumber, Representable};
use vm::{Py, PyObject, PyObjectRef, PyPayload, PyResult, VirtualMachine};

use pandas_rust_core::column::ColumnData;
use pandas_rust_core::ops::{aggregation, arithmetic, comparison, nulls, sort};
use pandas_rust_core::{DType, Series};

use crate::py_column::{
    agg_result_to_pyobj, column_to_pylist, column_value_to_pyobj, pandas_err, pyobj_to_scalar_value,
};

/// Python-visible Series class wrapping pandas_rust_core::Series.
#[vm::pyclass(module = "_pandas_native", name = "Series")]
#[derive(Debug, PyPayload)]
pub struct PySeries {
    pub(crate) inner: Series,
}

impl Clone for PySeries {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl PySeries {
    pub fn from_core(s: Series) -> Self {
        Self { inner: s }
    }

    pub fn to_py(self, vm: &VirtualMachine) -> PyObjectRef {
        self.into_pyobject(vm)
    }

    fn bin_arith(
        a: &PyObject,
        b: &PyObject,
        op_col: fn(
            &pandas_rust_core::Column,
            &pandas_rust_core::Column,
        ) -> pandas_rust_core::Result<pandas_rust_core::Column>,
        op_scalar: fn(
            &pandas_rust_core::Column,
            f64,
        ) -> pandas_rust_core::Result<pandas_rust_core::Column>,
        vm: &VirtualMachine,
    ) -> PyResult {
        let lhs = a
            .downcast_ref::<PySeries>()
            .ok_or_else(|| vm.new_type_error("left operand must be a Series".to_owned()))?;
        if let Some(rhs) = b.downcast_ref::<PySeries>() {
            let col =
                op_col(lhs.inner.column(), rhs.inner.column()).map_err(|e| pandas_err(e, vm))?;
            Ok(PySeries::from_core(Series::new(col)).into_pyobject(vm))
        } else {
            let scalar: f64 = b.to_owned().try_into_value(vm)?;
            let col = op_scalar(lhs.inner.column(), scalar).map_err(|e| pandas_err(e, vm))?;
            Ok(PySeries::from_core(Series::new(col)).into_pyobject(vm))
        }
    }
}

// Number protocol implementation for arithmetic operators
impl PySeries {
    const AS_NUMBER: PyNumberMethods = PyNumberMethods {
        add: Some(|a, b, vm| {
            PySeries::bin_arith(a, b, arithmetic::add, arithmetic::add_scalar, vm)
        }),
        subtract: Some(|a, b, vm| {
            PySeries::bin_arith(a, b, arithmetic::sub, arithmetic::sub_scalar, vm)
        }),
        multiply: Some(|a, b, vm| {
            PySeries::bin_arith(a, b, arithmetic::mul, arithmetic::mul_scalar, vm)
        }),
        true_divide: Some(|a, b, vm| {
            PySeries::bin_arith(a, b, arithmetic::div, arithmetic::div_scalar, vm)
        }),
        negative: Some(|a, vm| {
            let lhs = a
                .downcast_ref::<PySeries>()
                .ok_or_else(|| vm.new_type_error("operand must be a Series".to_owned()))?;
            let col = arithmetic::neg(lhs.inner.column()).map_err(|e| pandas_err(e, vm))?;
            Ok(PySeries::from_core(Series::new(col)).into_pyobject(vm))
        }),
        and: Some(|a, b, vm| {
            let lhs = a
                .downcast_ref::<PySeries>()
                .ok_or_else(|| vm.new_type_error("operand must be a Series".to_owned()))?;
            let rhs = b
                .downcast_ref::<PySeries>()
                .ok_or_else(|| vm.new_type_error("operand must be a Series".to_owned()))?;
            let col = comparison::and(lhs.inner.column(), rhs.inner.column())
                .map_err(|e| pandas_err(e, vm))?;
            Ok(PySeries::from_core(Series::new(col)).into_pyobject(vm))
        }),
        or: Some(|a, b, vm| {
            let lhs = a
                .downcast_ref::<PySeries>()
                .ok_or_else(|| vm.new_type_error("operand must be a Series".to_owned()))?;
            let rhs = b
                .downcast_ref::<PySeries>()
                .ok_or_else(|| vm.new_type_error("operand must be a Series".to_owned()))?;
            let col = comparison::or(lhs.inner.column(), rhs.inner.column())
                .map_err(|e| pandas_err(e, vm))?;
            Ok(PySeries::from_core(Series::new(col)).into_pyobject(vm))
        }),
        invert: Some(|a, vm| {
            let lhs = a
                .downcast_ref::<PySeries>()
                .ok_or_else(|| vm.new_type_error("operand must be a Series".to_owned()))?;
            let col = comparison::not(lhs.inner.column()).map_err(|e| pandas_err(e, vm))?;
            Ok(PySeries::from_core(Series::new(col)).into_pyobject(vm))
        }),
        ..PyNumberMethods::NOT_IMPLEMENTED
    };
}

impl AsNumber for PySeries {
    fn as_number() -> &'static PyNumberMethods {
        static AS_NUMBER: PyNumberMethods = PySeries::AS_NUMBER;
        &AS_NUMBER
    }
}

#[vm::pyclass(with(Representable, AsNumber))]
impl PySeries {
    #[pygetset]
    fn name(&self) -> String {
        self.inner.name().to_owned()
    }

    #[pygetset]
    fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    #[pymethod]
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[pymethod]
    fn __getitem__(&self, idx: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let i: i64 = idx.try_into_value(vm)?;
        let len = self.inner.len() as i64;
        let resolved = if i < 0 { len + i } else { i };
        if resolved < 0 || resolved as usize >= self.inner.len() {
            return Err(vm.new_index_error(format!(
                "index {} is out of bounds for Series of length {}",
                i,
                self.inner.len()
            )));
        }
        column_value_to_pyobj(self.inner.column(), resolved as usize, vm)
    }

    // --- Comparison (as regular methods since Comparable trait is complex) ---

    #[pymethod]
    fn eq(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PySeries> {
        if let Some(rhs) = other.downcast_ref::<PySeries>() {
            let col = comparison::eq(self.inner.column(), rhs.inner.column())
                .map_err(|e| pandas_err(e, vm))?;
            Ok(PySeries::from_core(Series::new(col)))
        } else {
            let scalar: f64 = other.try_into_value(vm)?;
            let col = comparison::eq_scalar(self.inner.column(), scalar)
                .map_err(|e| pandas_err(e, vm))?;
            Ok(PySeries::from_core(Series::new(col)))
        }
    }

    #[pymethod]
    fn ne(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PySeries> {
        if let Some(rhs) = other.downcast_ref::<PySeries>() {
            let col = comparison::ne(self.inner.column(), rhs.inner.column())
                .map_err(|e| pandas_err(e, vm))?;
            Ok(PySeries::from_core(Series::new(col)))
        } else {
            let eq_col = comparison::eq_scalar(self.inner.column(), other.try_into_value(vm)?)
                .map_err(|e| pandas_err(e, vm))?;
            let col = comparison::not(&eq_col).map_err(|e| pandas_err(e, vm))?;
            Ok(PySeries::from_core(Series::new(col)))
        }
    }

    #[pymethod]
    fn lt(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PySeries> {
        let rhs = other
            .downcast_ref::<PySeries>()
            .ok_or_else(|| vm.new_type_error("comparison requires a Series".to_owned()))?;
        let col = comparison::lt(self.inner.column(), rhs.inner.column())
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PySeries::from_core(Series::new(col)))
    }

    #[pymethod]
    fn le(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PySeries> {
        let rhs = other
            .downcast_ref::<PySeries>()
            .ok_or_else(|| vm.new_type_error("comparison requires a Series".to_owned()))?;
        let col = comparison::le(self.inner.column(), rhs.inner.column())
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PySeries::from_core(Series::new(col)))
    }

    #[pymethod]
    fn gt(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PySeries> {
        let rhs = other
            .downcast_ref::<PySeries>()
            .ok_or_else(|| vm.new_type_error("comparison requires a Series".to_owned()))?;
        let col = comparison::gt(self.inner.column(), rhs.inner.column())
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PySeries::from_core(Series::new(col)))
    }

    #[pymethod]
    fn ge(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PySeries> {
        let rhs = other
            .downcast_ref::<PySeries>()
            .ok_or_else(|| vm.new_type_error("comparison requires a Series".to_owned()))?;
        let col = comparison::ge(self.inner.column(), rhs.inner.column())
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PySeries::from_core(Series::new(col)))
    }

    // --- Aggregations ---

    #[pymethod]
    fn sum(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let r = aggregation::sum(self.inner.column()).map_err(|e| pandas_err(e, vm))?;
        Ok(agg_result_to_pyobj(&r, vm))
    }

    #[pymethod]
    fn mean(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let r = aggregation::mean(self.inner.column()).map_err(|e| pandas_err(e, vm))?;
        Ok(agg_result_to_pyobj(&r, vm))
    }

    #[pymethod]
    fn min(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let r = aggregation::min(self.inner.column()).map_err(|e| pandas_err(e, vm))?;
        Ok(agg_result_to_pyobj(&r, vm))
    }

    #[pymethod]
    fn max(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let r = aggregation::max(self.inner.column()).map_err(|e| pandas_err(e, vm))?;
        Ok(agg_result_to_pyobj(&r, vm))
    }

    #[pymethod]
    fn count(&self, vm: &VirtualMachine) -> PyObjectRef {
        let r = aggregation::count(self.inner.column());
        agg_result_to_pyobj(&r, vm)
    }

    #[pymethod]
    fn std(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let r = aggregation::std(self.inner.column(), 1).map_err(|e| pandas_err(e, vm))?;
        Ok(agg_result_to_pyobj(&r, vm))
    }

    #[pymethod]
    fn var(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let r = aggregation::var(self.inner.column(), 1).map_err(|e| pandas_err(e, vm))?;
        Ok(agg_result_to_pyobj(&r, vm))
    }

    #[pymethod]
    fn median(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let r = aggregation::median(self.inner.column()).map_err(|e| pandas_err(e, vm))?;
        Ok(agg_result_to_pyobj(&r, vm))
    }

    // --- Null operations ---

    #[pymethod]
    fn isna(&self) -> PySeries {
        let col = nulls::isna(self.inner.column());
        PySeries::from_core(Series::new(col))
    }

    #[pymethod]
    fn notna(&self) -> PySeries {
        let col = nulls::notna(self.inner.column());
        PySeries::from_core(Series::new(col))
    }

    #[pymethod]
    fn fillna(&self, value: PyObjectRef, vm: &VirtualMachine) -> PyResult<PySeries> {
        let sv = pyobj_to_scalar_value(&value, vm)?;
        let col = nulls::fillna_scalar(self.inner.column(), &sv).map_err(|e| pandas_err(e, vm))?;
        Ok(PySeries::from_core(Series::new(col)))
    }

    #[pymethod]
    fn dropna(&self) -> PySeries {
        let col = self.inner.column();
        let indices: Vec<usize> = (0..col.len()).filter(|&i| !col.is_null(i)).collect();
        let new_col = col.take(&indices).expect("dropna indices are always valid");
        PySeries::from_core(Series::new(new_col))
    }

    // --- Sort ---

    #[pymethod]
    fn sort_values(
        &self,
        ascending: vm::function::OptionalArg<bool>,
        _vm: &VirtualMachine,
    ) -> PySeries {
        let asc = ascending.unwrap_or(true);
        let indices = sort::argsort_column(self.inner.column(), asc);
        let new_col = self
            .inner
            .column()
            .take(&indices)
            .expect("argsort indices are always valid");
        PySeries::from_core(Series::new(new_col))
    }

    // --- Cast ---

    #[pymethod]
    fn astype(&self, dtype_str: PyObjectRef, vm: &VirtualMachine) -> PyResult<PySeries> {
        let s = dtype_str
            .downcast_ref::<PyStr>()
            .ok_or_else(|| vm.new_type_error("dtype must be a string".to_owned()))?;
        let target = DType::from_str(s.as_str())
            .ok_or_else(|| vm.new_value_error(format!("unknown dtype: {}", s.as_str())))?;
        let col = pandas_rust_core::casting::cast_column(self.inner.column(), target)
            .map_err(|e| pandas_err(e, vm))?;
        Ok(PySeries::from_core(Series::new(col)))
    }

    // --- Copy ---

    #[pymethod]
    fn copy(&self) -> PySeries {
        self.clone()
    }

    // --- Conversion ---

    #[pymethod]
    fn tolist(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        column_to_pylist(self.inner.column(), vm)
    }

    #[pymethod]
    fn to_dict(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let dict = vm.ctx.new_dict();
        for i in 0..self.inner.len() {
            let key: PyObjectRef = vm.ctx.new_int(i).into();
            let val = column_value_to_pyobj(self.inner.column(), i, vm)?;
            dict.set_item(&*key, val, vm)?;
        }
        Ok(dict.into())
    }

    #[pymethod]
    fn to_numpy(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        use numpy_rust_core::NdArray;
        use numpy_rust_python::py_array::PyNdArray;

        let col = self.inner.column();
        let arr = match col.data() {
            ColumnData::Bool(v) => {
                if col.has_nulls() {
                    // Promote to f64, nulls become NaN
                    let floats: Vec<f64> = v
                        .iter()
                        .enumerate()
                        .map(|(i, &b)| {
                            if col.is_null(i) { f64::NAN } else if b { 1.0 } else { 0.0 }
                        })
                        .collect();
                    NdArray::from_vec(floats)
                } else {
                    NdArray::from_vec(v.clone())
                }
            }
            ColumnData::Int64(v) => {
                if col.has_nulls() {
                    // Promote to f64, nulls become NaN
                    let floats: Vec<f64> = v
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| if col.is_null(i) { f64::NAN } else { x as f64 })
                        .collect();
                    NdArray::from_vec(floats)
                } else {
                    NdArray::from_vec(v.clone())
                }
            }
            ColumnData::Float64(v) => {
                if col.has_nulls() {
                    let floats: Vec<f64> = v
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| if col.is_null(i) { f64::NAN } else { x })
                        .collect();
                    NdArray::from_vec(floats)
                } else {
                    NdArray::from_vec(v.clone())
                }
            }
            ColumnData::Str(v) => {
                NdArray::from_vec(v.clone())
            }
        };
        Ok(PyNdArray::from_core(arr).into_pyobject(vm))
    }

    #[pygetset]
    fn values(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        self.to_numpy(vm)
    }
}

impl Representable for PySeries {
    fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
        let s = &zelf.inner;
        let col = s.column();
        let mut out = String::new();
        let max_show = 20;
        let len = col.len();
        let show = len.min(max_show);
        for i in 0..show {
            let val_str = if col.is_null(i) {
                "NaN".to_string()
            } else {
                match col.data() {
                    ColumnData::Bool(v) => format!("{}", v[i]),
                    ColumnData::Int64(v) => format!("{}", v[i]),
                    ColumnData::Float64(v) => format!("{}", v[i]),
                    ColumnData::Str(v) => v[i].clone(),
                }
            };
            out.push_str(&format!("{}    {}\n", i, val_str));
        }
        if len > max_show {
            out.push_str("...\n");
        }
        out.push_str(&format!(
            "Name: {}, Length: {}, dtype: {}",
            s.name(),
            len,
            s.dtype()
        ));
        Ok(out)
    }
}

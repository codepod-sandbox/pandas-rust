//! Internal conversion helpers between Python objects and pandas-rust-core Column types.
//! This module does NOT expose a Python class.

use rustpython_vm as vm;
use vm::builtins::{PyFloat, PyInt, PyList, PyStr};
use vm::{PyObjectRef, PyResult, VirtualMachine};

use pandas_rust_core::column::{Column, ColumnData};
use pandas_rust_core::ops::aggregation::AggResult;
use pandas_rust_core::ops::nulls::ScalarValue;
use pandas_rust_core::DType;

/// Convert a Python list to a Column, inferring dtype from the first non-None element.
pub fn pyobj_to_column(name: &str, data: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Column> {
    let list = data
        .downcast_ref::<PyList>()
        .ok_or_else(|| vm.new_type_error("expected a list".to_owned()))?;
    let items = list.borrow_vec();

    if items.is_empty() {
        return Ok(Column::new(name, ColumnData::Float64(vec![])));
    }

    // Find the first non-None element to infer dtype
    let mut inferred_dtype: Option<DType> = None;
    for item in items.iter() {
        if vm.is_none(item) {
            continue;
        }
        if item.downcast_ref::<vm::builtins::PyBool>().is_some() {
            inferred_dtype = Some(DType::Bool);
        } else if item.downcast_ref::<PyInt>().is_some() {
            inferred_dtype = Some(DType::Int64);
        } else if item.downcast_ref::<PyFloat>().is_some() {
            inferred_dtype = Some(DType::Float64);
        } else if item.downcast_ref::<PyStr>().is_some() {
            inferred_dtype = Some(DType::Str);
        } else {
            return Err(vm.new_type_error(format!(
                "unsupported element type in list for column '{}'",
                name
            )));
        }
        break;
    }

    let dtype = inferred_dtype.unwrap_or(DType::Float64);

    // Build null mask and data
    let mut null_mask = vec![false; items.len()];
    let mut has_nulls = false;

    let col = match dtype {
        DType::Bool => {
            let mut vals = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                if vm.is_none(item) {
                    vals.push(false);
                    null_mask[i] = true;
                    has_nulls = true;
                } else {
                    let b: bool = item.clone().try_into_value(vm)?;
                    vals.push(b);
                }
            }
            if has_nulls {
                Column::new_with_nulls(name, ColumnData::Bool(vals), null_mask)
                    .map_err(|e| vm.new_value_error(e.to_string()))?
            } else {
                Column::new(name, ColumnData::Bool(vals))
            }
        }
        DType::Int64 => {
            let mut vals = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                if vm.is_none(item) {
                    vals.push(0i64);
                    null_mask[i] = true;
                    has_nulls = true;
                } else {
                    let v: i64 = item.clone().try_into_value(vm)?;
                    vals.push(v);
                }
            }
            if has_nulls {
                Column::new_with_nulls(name, ColumnData::Int64(vals), null_mask)
                    .map_err(|e| vm.new_value_error(e.to_string()))?
            } else {
                Column::new(name, ColumnData::Int64(vals))
            }
        }
        DType::Float64 => {
            let mut vals = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                if vm.is_none(item) {
                    vals.push(f64::NAN);
                    null_mask[i] = true;
                    has_nulls = true;
                } else {
                    let v: f64 = item.clone().try_into_value(vm)?;
                    vals.push(v);
                }
            }
            if has_nulls {
                Column::new_with_nulls(name, ColumnData::Float64(vals), null_mask)
                    .map_err(|e| vm.new_value_error(e.to_string()))?
            } else {
                Column::new(name, ColumnData::Float64(vals))
            }
        }
        DType::Str => {
            let mut vals = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                if vm.is_none(item) {
                    vals.push(String::new());
                    null_mask[i] = true;
                    has_nulls = true;
                } else {
                    let s = item
                        .downcast_ref::<PyStr>()
                        .ok_or_else(|| vm.new_type_error("expected string element".to_owned()))?;
                    vals.push(s.as_str().to_owned());
                }
            }
            if has_nulls {
                Column::new_with_nulls(name, ColumnData::Str(vals), null_mask)
                    .map_err(|e| vm.new_value_error(e.to_string()))?
            } else {
                Column::new(name, ColumnData::Str(vals))
            }
        }
    };

    Ok(col)
}

/// Extract a single value from a column at the given index as a Python object.
/// Null values are returned as Python None.
pub fn column_value_to_pyobj(
    col: &Column,
    idx: usize,
    vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    if col.is_null(idx) {
        return Ok(vm.ctx.none());
    }
    match col.data() {
        ColumnData::Bool(v) => Ok(vm.ctx.new_bool(v[idx]).into()),
        ColumnData::Int64(v) => Ok(vm.ctx.new_int(v[idx]).into()),
        ColumnData::Float64(v) => Ok(vm.ctx.new_float(v[idx]).into()),
        ColumnData::Str(v) => Ok(vm.ctx.new_str(v[idx].as_str()).into()),
    }
}

/// Convert an entire column to a Python list.
pub fn column_to_pylist(col: &Column, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    let mut items = Vec::with_capacity(col.len());
    for i in 0..col.len() {
        items.push(column_value_to_pyobj(col, i, vm)?);
    }
    Ok(vm.ctx.new_list(items).into())
}

/// Convert an AggResult to a Python scalar.
pub fn agg_result_to_pyobj(result: &AggResult, vm: &VirtualMachine) -> PyObjectRef {
    match result {
        AggResult::Int64(v) => vm.ctx.new_int(*v).into(),
        AggResult::Float64(v) => vm.ctx.new_float(*v).into(),
        AggResult::Str(v) => vm.ctx.new_str(v.as_str()).into(),
        AggResult::Usize(v) => vm.ctx.new_int(*v).into(),
        AggResult::None => vm.ctx.none(),
    }
}

/// Convert a Python object to a ScalarValue for fillna operations.
pub fn pyobj_to_scalar_value(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<ScalarValue> {
    if obj.downcast_ref::<vm::builtins::PyBool>().is_some() {
        let b: bool = obj.clone().try_into_value(vm)?;
        return Ok(ScalarValue::Bool(b));
    }
    if obj.downcast_ref::<PyInt>().is_some() {
        let v: i64 = obj.clone().try_into_value(vm)?;
        return Ok(ScalarValue::Int64(v));
    }
    if obj.downcast_ref::<PyFloat>().is_some() {
        let v: f64 = obj.clone().try_into_value(vm)?;
        return Ok(ScalarValue::Float64(v));
    }
    if let Some(s) = obj.downcast_ref::<PyStr>() {
        return Ok(ScalarValue::Str(s.as_str().to_owned()));
    }
    Err(vm.new_type_error("fillna value must be bool, int, float, or str".to_owned()))
}

/// Helper to convert a pandas-rust error to a Python exception.
pub fn pandas_err(
    e: pandas_rust_core::PandasError,
    vm: &VirtualMachine,
) -> vm::PyRef<vm::builtins::PyBaseException> {
    match e {
        pandas_rust_core::PandasError::KeyError(msg) => {
            vm.new_key_error(vm.ctx.new_str(msg).into())
        }
        pandas_rust_core::PandasError::IndexError(msg) => vm.new_index_error(msg),
        pandas_rust_core::PandasError::TypeError(msg) => vm.new_type_error(msg),
        pandas_rust_core::PandasError::ValueError(msg) => vm.new_value_error(msg),
        pandas_rust_core::PandasError::IoError(e) => vm.new_value_error(e.to_string()),
        pandas_rust_core::PandasError::ParseError(msg) => vm.new_value_error(msg),
    }
}

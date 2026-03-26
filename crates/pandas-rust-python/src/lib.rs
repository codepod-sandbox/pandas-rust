pub mod py_column;
pub mod py_dataframe;
pub mod py_groupby;
pub mod py_index;
pub mod py_io;
pub mod py_series;

use rustpython_vm as vm;

/// Return the native pandas module definition for registration with the interpreter builder.
pub fn pandas_module_def(ctx: &vm::Context) -> &'static vm::builtins::PyModuleDef {
    _pandas_native::module_def(ctx)
}

#[vm::pymodule]
pub mod _pandas_native {
    use super::*;
    use crate::py_dataframe::PyDataFrame;
    use crate::py_groupby::PyGroupBy;
    use crate::py_index::PyIndex;
    use crate::py_series::PySeries;
    use vm::builtins::PyStr;
    use vm::class::PyClassImpl;
    use vm::{PyObjectRef, PyRef, PyResult, VirtualMachine};

    // Register class types as module attributes
    #[allow(non_snake_case)]
    #[pyattr]
    fn DataFrame(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PyDataFrame::make_class(&vm.ctx)
    }

    #[allow(non_snake_case)]
    #[pyattr]
    fn Series(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PySeries::make_class(&vm.ctx)
    }

    #[allow(non_snake_case)]
    #[pyattr]
    fn Index(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PyIndex::make_class(&vm.ctx)
    }

    #[allow(non_snake_case)]
    #[pyattr]
    fn GroupBy(vm: &VirtualMachine) -> vm::builtins::PyTypeRef {
        PyGroupBy::make_class(&vm.ctx)
    }

    #[pyfunction]
    fn read_csv(
        path: PyRef<PyStr>,
        delimiter: vm::function::OptionalArg<PyRef<PyStr>>,
        header: vm::function::OptionalArg<bool>,
        vm: &VirtualMachine,
    ) -> PyResult<PyDataFrame> {
        py_io::py_read_csv(path, delimiter, header, vm)
    }

    #[pyfunction]
    fn _version(_vm: &VirtualMachine) -> PyResult<String> {
        Ok("0.1.0".to_string())
    }

    #[pyfunction]
    fn concat(
        objs: PyObjectRef,
        axis: vm::function::OptionalArg<i32>,
        vm: &VirtualMachine,
    ) -> PyResult<PyDataFrame> {
        use pandas_rust_core::concat as core_concat;
        use pandas_rust_core::DataFrame;
        let axis = axis.unwrap_or(0);
        let list = objs.downcast_ref::<vm::builtins::PyList>().ok_or_else(|| {
            vm.new_type_error("concat() requires a list of DataFrames".to_owned())
        })?;
        let items = list.borrow_vec();
        let dfs: PyResult<Vec<DataFrame>> = items
            .iter()
            .map(|obj| {
                obj.downcast_ref::<PyDataFrame>()
                    .ok_or_else(|| {
                        vm.new_type_error("concat() list items must be DataFrames".to_owned())
                    })
                    .map(|pdf| pdf.data.read().unwrap().clone())
            })
            .collect();
        let dfs = dfs?;
        let refs: Vec<&DataFrame> = dfs.iter().collect();
        let result = if axis == 0 {
            core_concat::concat_rows(&refs)
        } else {
            core_concat::concat_cols(&refs)
        }
        .map_err(|e| crate::py_column::pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(result))
    }
}

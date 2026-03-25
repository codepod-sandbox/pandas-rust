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
    use vm::{PyRef, PyResult, VirtualMachine};

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
}

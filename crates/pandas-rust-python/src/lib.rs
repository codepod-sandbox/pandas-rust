use rustpython_vm as vm;

/// Return the native pandas module definition for registration with the interpreter builder.
pub fn pandas_module_def(ctx: &vm::Context) -> &'static vm::builtins::PyModuleDef {
    _pandas_native::module_def(ctx)
}

#[vm::pymodule]
pub mod _pandas_native {
    use rustpython_vm as vm;
    use vm::PyResult;
    use vm::VirtualMachine;

    #[pyfunction]
    fn _version(_vm: &VirtualMachine) -> PyResult<String> {
        Ok("0.1.0".to_string())
    }
}

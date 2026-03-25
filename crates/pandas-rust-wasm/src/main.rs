use std::path::PathBuf;
use std::process::ExitCode;

use rustpython::{InterpreterBuilder, InterpreterBuilderExt};

pub fn main() -> ExitCode {
    let root = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent()?.parent()?.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    let paths = [root.join("python"), root.join("packages/numpy-rust/python")];
    let prepend = paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(":");
    let pythonpath = match std::env::var("PYTHONPATH") {
        Ok(existing) => format!("{}:{}", prepend, existing),
        Err(_) => prepend,
    };
    std::env::set_var("PYTHONPATH", pythonpath);

    let config = InterpreterBuilder::new().init_stdlib();

    let numpy_def = numpy_rust_python::numpy_module_def(&config.ctx);
    let config = config.add_native_module(numpy_def);

    let pandas_def = pandas_rust_python::pandas_module_def(&config.ctx);
    let config = config.add_native_module(pandas_def);

    rustpython::run(config)
}

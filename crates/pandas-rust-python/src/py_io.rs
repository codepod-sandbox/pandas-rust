use rustpython_vm as vm;
use vm::builtins::PyStr;
use vm::{PyRef, PyResult, VirtualMachine};

use pandas_rust_core::io::csv::{read_csv, CsvReadOptions};

use crate::py_column::pandas_err;
use crate::py_dataframe::PyDataFrame;

/// read_csv(path, delimiter=",", header=True) -> DataFrame
pub fn py_read_csv(
    path: PyRef<PyStr>,
    delimiter: vm::function::OptionalArg<PyRef<PyStr>>,
    header: vm::function::OptionalArg<bool>,
    vm: &VirtualMachine,
) -> PyResult<PyDataFrame> {
    let mut options = CsvReadOptions::default();

    if let Some(d) = delimiter.into_option() {
        let d_str = d.as_str();
        if d_str.len() != 1 {
            return Err(vm.new_value_error("delimiter must be a single character".to_owned()));
        }
        options.delimiter = d_str.as_bytes()[0];
    }

    if let Some(h) = header.into_option() {
        options.has_header = h;
    }

    let file = std::fs::File::open(path.as_str())
        .map_err(|e| vm.new_value_error(format!("cannot open file '{}': {}", path.as_str(), e)))?;
    let reader = std::io::BufReader::new(file);
    let df = read_csv(reader, options).map_err(|e| pandas_err(e, vm))?;
    Ok(PyDataFrame::from_core(df))
}

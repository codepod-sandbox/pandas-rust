use rustpython_vm as vm;
use vm::types::Representable;
use vm::{Py, PyPayload, PyResult, VirtualMachine};

use pandas_rust_core::groupby::{self, AggFn, GroupedData};
use pandas_rust_core::DataFrame;

use crate::py_column::pandas_err;
use crate::py_dataframe::PyDataFrame;

/// Python-visible GroupBy object.
#[vm::pyclass(module = "_pandas_native", name = "GroupBy")]
#[derive(Debug, PyPayload)]
pub struct PyGroupBy {
    df: DataFrame,
    by_columns: Vec<String>,
}

impl PyGroupBy {
    pub fn new(df: DataFrame, by_columns: Vec<String>) -> Self {
        Self { df, by_columns }
    }

    fn grouped(&self, vm: &VirtualMachine) -> PyResult<GroupedData> {
        let by_refs: Vec<&str> = self.by_columns.iter().map(|s| s.as_str()).collect();
        groupby::group_by(&self.df, &by_refs).map_err(|e| pandas_err(e, vm))
    }

    fn agg_method(&self, agg_fn: AggFn, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        let grouped = self.grouped(vm)?;
        let result =
            groupby::aggregate(&grouped, &self.df, agg_fn).map_err(|e| pandas_err(e, vm))?;
        Ok(PyDataFrame::from_core(result))
    }
}

#[vm::pyclass(with(Representable))]
impl PyGroupBy {
    #[pymethod]
    fn sum(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::Sum, vm)
    }

    #[pymethod]
    fn mean(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::Mean, vm)
    }

    #[pymethod]
    fn min(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::Min, vm)
    }

    #[pymethod]
    fn max(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::Max, vm)
    }

    #[pymethod]
    fn count(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::Count, vm)
    }

    #[pymethod]
    fn std(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::Std, vm)
    }

    #[pymethod]
    fn var(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::Var, vm)
    }

    #[pymethod]
    fn median(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::Median, vm)
    }

    #[pymethod]
    fn first(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::First, vm)
    }

    #[pymethod]
    fn last(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::Last, vm)
    }

    #[pymethod]
    fn size(&self, vm: &VirtualMachine) -> PyResult<PyDataFrame> {
        self.agg_method(AggFn::Size, vm)
    }
}

impl Representable for PyGroupBy {
    fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
        Ok(format!("<GroupBy [{}]>", zelf.by_columns.join(", ")))
    }
}

use rustpython_vm as vm;
use vm::types::Representable;
use vm::{Py, PyObjectRef, PyPayload, PyResult, VirtualMachine};

use pandas_rust_core::index::Index;

/// Python-visible Index wrapper.
#[vm::pyclass(module = "_pandas_native", name = "Index")]
#[derive(Debug, PyPayload)]
pub struct PyIndex {
    pub(crate) inner: Index,
}

impl Clone for PyIndex {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl PyIndex {
    pub fn from_core(idx: Index) -> Self {
        Self { inner: idx }
    }
}

#[vm::pyclass(with(Representable))]
impl PyIndex {
    #[pymethod]
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[pymethod]
    fn tolist(&self, vm: &VirtualMachine) -> PyObjectRef {
        let items: Vec<PyObjectRef> = match &self.inner {
            Index::Range(r) => (0..r.len())
                .map(|i| {
                    let val = r.get(i).unwrap();
                    vm.ctx.new_int(val).into()
                })
                .collect(),
            Index::Int64(v) => v.iter().map(|&val| vm.ctx.new_int(val).into()).collect(),
            Index::Str(v) => v
                .iter()
                .map(|val| vm.ctx.new_str(val.as_str()).into())
                .collect(),
        };
        vm.ctx.new_list(items).into()
    }
}

impl Representable for PyIndex {
    fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
        match &zelf.inner {
            Index::Range(r) => Ok(format!(
                "RangeIndex(start={}, stop={}, step={})",
                r.start, r.stop, r.step
            )),
            Index::Int64(v) => {
                let vals: Vec<String> = v.iter().map(|x| x.to_string()).collect();
                Ok(format!("Int64Index([{}])", vals.join(", ")))
            }
            Index::Str(v) => {
                let vals: Vec<String> = v.iter().map(|x| format!("'{}'", x)).collect();
                Ok(format!("Index([{}])", vals.join(", ")))
            }
        }
    }
}

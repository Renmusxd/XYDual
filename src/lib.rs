mod lattice;

use crate::lattice::Lattice;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

#[pyclass]
struct PyLattice {
    lat: Lattice,
}
#[pymethods]
impl PyLattice {
    #[new]
    fn new(lx: usize, ly: usize, pots: Vec<f64>) -> Self {
        Self {
            lat: Lattice::new(lx, ly, pots),
        }
    }

    fn update(&mut self, updates: Option<usize>) {
        let updates = updates.unwrap_or(1);
        for _ in 0..updates {
            self.lat.update()
        }
    }

    fn get_energy(&self) -> f64 {
        self.lat.get_total_energy()
    }

    fn simulate_and_get_energy(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
    ) -> Py<PyArray1<f64>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);

        let mut energy = Array1::<f64>::zeros((num_samples,));
        energy.iter_mut().for_each(|x| {
            for _ in 0..local_updates_per_step {
                self.lat.update();
            }
            *x = self.lat.get_total_energy();
        });
        energy.into_pyarray(py).to_owned()
    }
}

#[pymodule]
fn py_xydual(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLattice>()?;
    Ok(())
}

mod lattice;

use crate::lattice::Lattice;
use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
struct PyLattice {
    lat: Vec<Lattice>,
}
#[pymethods]
impl PyLattice {
    #[new]
    fn new(
        lx: usize,
        ly: usize,
        pots: Vec<f64>,
        experiments: Option<usize>,
        boundary: Option<(i32, i32)>,
    ) -> Self {
        let experiments = experiments.unwrap_or(1);
        let boundary = boundary.unwrap_or((0, 0));
        Self {
            lat: (0..experiments)
                .map(|_| Lattice::new(lx, ly, pots.clone(), boundary))
                .collect(),
        }
    }

    fn update(&mut self, updates: Option<usize>) {
        let updates = updates.unwrap_or(1);

        self.lat.par_iter_mut().for_each(|lat| {
            let mut rng = rand::thread_rng();
            for _ in 0..updates {
                lat.new_update(&mut rng)
            }
        });
    }

    fn get_energy(&self, py: Python) -> Py<PyArray1<f64>> {
        let mut energies = Array1::zeros((self.lat.len(),));
        energies
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(self.lat.par_iter())
            .for_each(|(mut e, lat)| *e.get_mut([]).unwrap() = lat.get_total_energy());
        energies.into_pyarray(py).to_owned()
    }

    fn simulate_and_get_energy(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
    ) -> Py<PyArray2<f64>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);

        let mut energy = Array2::<f64>::zeros((num_samples, self.lat.len()));
        energy.axis_iter_mut(Axis(0)).for_each(|mut x| {
            x.axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(self.lat.par_iter_mut())
                .for_each(|(mut e, lat)| {
                    let mut rng = rand::thread_rng();
                    for _ in 0..local_updates_per_step {
                        lat.new_update(&mut rng)
                    }
                    *e.get_mut([]).unwrap() = lat.get_total_energy();
                });
        });
        energy.into_pyarray(py).to_owned()
    }
}

#[pymodule]
fn py_xydual(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLattice>()?;
    Ok(())
}

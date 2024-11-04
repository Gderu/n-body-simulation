use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;

const MIN_DIST: f64 = 0.5;
const G: f64 = 6.67428e-11;


/// Formats the sum of two numbers as string.
#[pyfunction]
fn calc_acceleration_brute_force(positions: PyReadonlyArray2<'_, f64>,
                                 masses: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyArray2<f64>>> {
    let pos_array = positions.as_array();
    let masses_array = masses.as_array();
    let accelerations: Vec<Vec<f64>> = pos_array.outer_iter().collect::<Vec<_>>().par_iter().
        enumerate().map(|(i, cur_pos)| {
        let mut acceleration = vec![0.; 3];
        for (p, m) in pos_array.outer_iter().zip(masses_array.iter()) {
            let diff  = [p[0] - cur_pos[0], p[1] - cur_pos[1], p[2] - cur_pos[2]];
            if diff != [0.; 3] {
                let dist_sqr = diff.iter().map(|x| x * x).sum::<f64>().max(MIN_DIST);
                let mult_by = G * m * dist_sqr.powf(-1.5);
                for j in 0..3 {
                    acceleration[j] += diff[j] * mult_by;
                }
            }
        }
        acceleration
    }).collect();
    Python::with_gil(|py| {
        let array = PyArray2::from_vec2_bound(py, &accelerations)?;
        Ok(array.unbind())
    })
}


/// A Python module implemented in Rust.
#[pymodule]
fn star_simulation_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_acceleration_brute_force, m)?)?;
    Ok(())
}

use std::array;
use std::ops::Div;
use std::time::Instant;
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{s, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use itertools::Itertools;
use rayon::ThreadPoolBuilder;

const MIN_DIST: f64 = 1.392e9;
const G: f64 = 6.67428e-11;


//an octree node
enum Node {
    Leaf { pos: [f64; 3], mass: f64 },
    Branch { children: Box<[Option<Node>; 8]>, mass: f64, center_of_mass: [f64; 3], width: f64 }
}


// Calculates the acceleration acting on each body individually without barnes-hut
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

#[pyfunction]
fn test(positions: PyReadonlyArray2<'_, f64>, masses: PyReadonlyArray1<'_, f64>) {
    let pos_array = positions.as_array();
    let masses_array = masses.as_array();
    let start = Instant::now();
    let mut bounds = (0., 0.);
    for _ in 0..100 {
        bounds = get_bounds(pos_array);
    }
    println!("get bounds: {:?}", start.elapsed() / 100);
    let start = Instant::now();
    for _ in 0..100 {
        let morton_codes = convert_to_morton_code(pos_array, &bounds);
    }
    println!("morton codes: {:?}", start.elapsed() / 100);
}

#[pyfunction]
fn calc_acceleration_barnes_hut(positions: PyReadonlyArray2<'_, f64>,
                                masses: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyArray2<f64>>> {
    let pos_array = positions.as_array();
    let masses_array = masses.as_array();
    let bounds = get_bounds(pos_array);
    let morton_code = convert_to_morton_code(pos_array, &bounds);
    // let octree = populate_octree(pos_array, masses_array);

    Python::with_gil(|py| {
        let array = PyArray2::from_vec2_bound(py, &vec![])?;
        Ok(array.unbind())
    })
}


fn get_bounds(positions: ArrayView2<f64>) -> (f64, f64) {
    let num_threads = 8;
    let chunk_size = positions.shape()[0].div_ceil(num_threads);
    (0..num_threads).into_par_iter().map(|i| {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(positions.shape()[0]);

        if start > end {
            return (f64::INFINITY, f64::NEG_INFINITY);
        }

        positions.slice(s![start..end, ..]).iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY),
                  |(min, max), &x| (min.min(x), max.max(x)))
    }).reduce(|| (f64::INFINITY, f64::NEG_INFINITY),
              |x, y| (x.0.min(y.0), x.1.max(y.1)))
}


// converting the positions in floating point to be in a 1e9 box in u32 coords,
// then converting those coords to morton code, and sorting the resulting array
fn convert_to_morton_code(positions: ArrayView2<f64>, bounds: &(f64, f64)) -> Vec<(u128, usize)> {
    let diff = (2u64.pow(32) - 1) as f64 / (bounds.1 - bounds.0);

    let mut morton_codes: Vec<(u128, usize)> = vec![(0, 0); positions.shape()[0]];
    morton_codes.par_iter_mut().enumerate().for_each(|(i, elem)| {
        *elem = (morton_encode([((positions[[i, 0]] - bounds.0) * diff) as u32,
            ((positions[[i, 1]] - bounds.0) *  diff) as u32,
            ((positions[[i, 2]] - bounds.0) *  diff) as u32]), i);
    });

    morton_codes.par_sort_by(|a , b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Greater));
    morton_codes
}


/// Interleaves the bits of a 3D point `[x, y, z]` into a 3D Morton code.
/// Each coordinate is a `u32`, and the output is a `u64` Morton code.
fn morton_encode(coords: [u32; 3]) -> u128 {
    let (mut x, mut y, mut z) = (coords[0] as u128, coords[1] as u128, coords[2] as u128);

    x &= 0x3ffffffffff;
    x = (x | x << 64) & 0x3ff0000000000000000ffffffff;
    x = (x | x << 32) & 0x3ff00000000ffff00000000ffff;
    x = (x | x << 16) & 0x30000ff0000ff0000ff0000ff0000ff;
    x = (x | x << 8) & 0x300f00f00f00f00f00f00f00f00f00f;
    x = (x | x << 4) & 0x30c30c30c30c30c30c30c30c30c30c3;
    x = (x | x << 2) & 0x9249249249249249249249249249249;

    y &= 0x3ffffffffff;
    y = (y | y << 64) & 0x3ff0000000000000000ffffffff;
    y = (y | y << 32) & 0x3ff00000000ffff00000000ffff;
    y = (y | y << 16) & 0x30000ff0000ff0000ff0000ff0000ff;
    y = (y | y << 8) & 0x300f00f00f00f00f00f00f00f00f00f;
    y = (y | y << 4) & 0x30c30c30c30c30c30c30c30c30c30c3;
    y = (y | y << 2) & 0x9249249249249249249249249249249;

    z &= 0x3ffffffffff;
    z = (z | z << 64) & 0x3ff0000000000000000ffffffff;
    z = (z | z << 32) & 0x3ff00000000ffff00000000ffff;
    z = (z | z << 16) & 0x30000ff0000ff0000ff0000ff0000ff;
    z = (z | z << 8) & 0x300f00f00f00f00f00f00f00f00f00f;
    z = (z | z << 4) & 0x30c30c30c30c30c30c30c30c30c30c3;
    z = (z | z << 2) & 0x9249249249249249249249249249249;

    // Combine the interleaved bits of x, y, and z
    (x << 2) | (y << 1) | z
}


// fn create_tree(bounds: Vec<(f64, f64)>, morton_code: Vec<(u128, usize)>, positions: ArrayView2<f64>, masses: ArrayView1<f64>) -> Node {
//     let pool = ThreadPoolBuilder::new().num_threads(40).build().unwrap();
//     pool.install(move || {
//         let _ = morton_code.partition_point(|&x| x.0 < );
//         let root = Node::Branch {
//             children: Box::new([None; 8]),
//             mass: 0.0,
//             center_of_mass: [0.; 3],
//             width: 0.0,
//         };
//
//         root
//     })
// }

// fn populate_octree(positions: ArrayView2<f64>, masses: ArrayView1<f64>) -> Node {
//     let mut root = Node::Empty;
//     // a vector of length 3 with each element being a pair of minimum and maximum values for that coord
//     let mut bounds = get_bounds(positions);
//
//     for (new_pos, &new_mass) in positions.outer_iter().zip(masses.iter()) {
//         let mut cur_node = &mut root;
//         loop {
//             let mut idx = None;
//             match &mut cur_node {
//                 Node::Empty => {
//                     *cur_node = Node::Leaf { pos: [new_pos[0], new_pos[1], new_pos[2]], mass: new_mass};
//                     break;
//                 },
//                 Node::Leaf { pos, mass} => {
//                     let mut children = array::from_fn(|_| Node::Empty);
//                     let idx = get_octant(&bounds, pos);
//                     children[idx] = Node::Leaf { pos: *pos, mass: *mass};
//
//                     *cur_node = Node::Branch {
//                         children: Box::new(children),
//                         mass: *mass,
//                         avg_pos: *pos,
//                         num_inside: 2
//                     };
//                 },
//                 Node::Branch { children, mass, avg_pos, num_inside} => {
//                     *mass += new_mass;
//                     for i in 0..3 {
//                         avg_pos[i] = (avg_pos[i] * (*num_inside as f64) + new_pos[i]) / (*num_inside as f64 + 1.);
//                     }
//                     idx = Some(get_octant(&bounds, &[new_pos[0], new_pos[1], new_pos[2]]));
//                 }
//             };
//             if let Some(idx) = idx {
//                 for i in 0..3 {
//
//                 }
//             }
//         }
//     }
//     Node::Empty
// }

/// A Python module implemented in Rust.
#[pymodule]
fn star_simulation_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_acceleration_brute_force, m)?)?;
    m.add_function(wrap_pyfunction!(calc_acceleration_barnes_hut, m)?)?;
    m.add_function(wrap_pyfunction!(test, m)?)?;
    Ok(())
}

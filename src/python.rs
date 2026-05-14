use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::router::{phase_router, phase_router_uniform_dispatch};


/// Low-level binding: thin wrapper over the Rust kernel.
///
/// Accepts flat NumPy arrays, releases GIL during compute,
/// returns a 2D (n, k) NumPy array of routed column indices (-1 = empty).
#[pyfunction]
#[pyo3(name = "phase_router")]
fn phase_router_py<'py>(
    py: Python<'py>,
    s_bits: PyReadonlyArray1<'py, u64>,
    t_bits: PyReadonlyArray1<'py, u64>,
    n: usize,
    nb_words: usize,
    k: usize,
    col_perm_s: PyReadonlyArray1<'py, usize>,
    col_perm_t: PyReadonlyArray1<'py, usize>,
    seed: u64,
) -> PyResult<&'py PyArray2<i32>> {
    let s_bits = s_bits.as_slice()?;
    let t_bits = t_bits.as_slice()?;
    let col_perm_s = col_perm_s.as_slice()?;
    let col_perm_t = col_perm_t.as_slice()?;

    // Release GIL during compute — critical for Rayon parallelism
    let routes = py.allow_threads(|| {
        phase_router(s_bits, t_bits, n, nb_words, k, col_perm_s, col_perm_t, seed)
    });

    // Return as 2D (n, k) array for ergonomic Python use
    let arr = Array2::from_shape_vec((n, k), routes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// High-level binding: generates permutations internally.
///
/// Users only provide bit-packed matrices, dimensions, fan-out, and seed.
/// Permutations are derived deterministically from the seed.
#[pyfunction]
#[pyo3(name = "phase_router_auto")]
fn phase_router_auto_py<'py>(
    py: Python<'py>,
    s_bits: PyReadonlyArray1<'py, u64>,
    t_bits: PyReadonlyArray1<'py, u64>,
    n: usize,
    k: usize,
    seed: u64,
) -> PyResult<&'py PyArray2<i32>> {
    let s_bits = s_bits.as_slice()?;
    let t_bits = t_bits.as_slice()?;
    let nb_words = (n + 63) / 64;

    // Generate permutations deterministically from seed
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut col_perm_s: Vec<usize> = (0..n).collect();
    let mut col_perm_t: Vec<usize> = (0..n).collect();
    col_perm_s.shuffle(&mut rng);
    col_perm_t.shuffle(&mut rng);

    // Release GIL during compute
    let routes = py.allow_threads(|| {
        phase_router(
            s_bits,
            t_bits,
            n,
            nb_words,
            k,
            &col_perm_s,
            &col_perm_t,
            seed,
        )
    });

    let arr = Array2::from_shape_vec((n, k), routes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Fused MoE dispatch binding: builds (uniform) s_bits/t_bits, runs the
/// phase-routing kernel, maps kernel columns to expert ids via band tiling,
/// and dedupes the first `k` unique experts per token — all in a single
/// GIL-released Rust call.
///
/// Returns an `(n_tokens, k)` int32 NumPy array of expert ids in
/// `[0, n_experts)`, with `-1` for unfilled slots.
#[pyfunction]
#[pyo3(
    name = "phase_router_uniform_dispatch",
    signature = (n_tokens, n_experts, k, base_density, seed, oversample = 4),
)]
fn phase_router_uniform_dispatch_py<'py>(
    py: Python<'py>,
    n_tokens: usize,
    n_experts: usize,
    k: usize,
    base_density: f64,
    seed: u64,
    oversample: usize,
) -> PyResult<&'py PyArray2<i32>> {
    if n_tokens == 0 || n_experts == 0 || k == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_tokens, n_experts, k must all be > 0",
        ));
    }
    if oversample == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "oversample must be > 0",
        ));
    }

    let routes = py.allow_threads(|| {
        phase_router_uniform_dispatch(n_tokens, n_experts, k, base_density, oversample, seed)
    });

    let arr = Array2::from_shape_vec((n_tokens, k), routes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Python module definition.
#[pymodule]
fn phase_router_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(phase_router_py, m)?)?;
    m.add_function(wrap_pyfunction!(phase_router_auto_py, m)?)?;
    m.add_function(wrap_pyfunction!(phase_router_uniform_dispatch_py, m)?)?;
    Ok(())
}


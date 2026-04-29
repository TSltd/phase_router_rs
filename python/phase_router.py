"""
High-level Python API for the Phase Router.

Wraps the Rust kernel — handles bit-packing, permutation generation,
and result formatting so users never touch u64 arrays directly.

Usage:
    from phase_router import route, build_bits

    routes = route(
        source_weights=[1]*1024,
        target_capacities=[1.0, 1.0, 8.0, ...],
        k=4,
        seed=42,
        density=0.3,
    )
    # routes: np.ndarray shape (n, k), dtype int32
"""

import numpy as np

try:
    import phase_router_rs as _rs
except ImportError:
    _rs = None


# ── Bit-packing ──────────────────────────────────────────────────────────

def build_bits(ones_per_row: np.ndarray, n: int) -> np.ndarray:
    """
    Pack a bit matrix: row i has ones_per_row[i] consecutive bits set
    starting at position 0 (left-aligned).

    Returns a flat uint64 array of shape (n * nb_words,).
    """
    nb_words = (n + 63) // 64
    bits = np.zeros(n * nb_words, dtype=np.uint64)

    for i in range(n):
        ones = int(min(ones_per_row[i], n))
        for b in range(ones):
            word = b // 64
            bit = b % 64
            bits[i * nb_words + word] |= np.uint64(1) << np.uint64(bit)

    return bits


# ── Hash routing baseline ────────────────────────────────────────────────

def hash_route(n: int, k: int, seed: int = 0) -> np.ndarray:
    """
    Uniform hash routing: each source picks k targets uniformly at random.
    Returns (n, k) array of target indices.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n, k), dtype=np.int32)


# ── Capacity enforcement ─────────────────────────────────────────────────

def enforce_capacity(routes: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """
    Apply hard capacity limits. Tokens exceeding an expert's cap are dropped
    (set to -1). Returns a copy with drops applied.

    Parameters:
        routes: (n, k) int32 array of target indices (-1 = empty)
        caps:   (n,) int array of per-target capacity limits
    """
    out = routes.copy()
    n = len(caps)
    loads = np.zeros(n, dtype=np.int64)

    flat = out.ravel()
    for idx in range(len(flat)):
        t = flat[idx]
        if 0 <= t < n:
            if loads[t] < caps[t]:
                loads[t] += 1
            else:
                flat[idx] = -1

    return out


# ── Load analysis ────────────────────────────────────────────────────────

def compute_loads(routes: np.ndarray, n: int) -> np.ndarray:
    """Count how many tokens are routed to each target (ignoring -1s)."""
    flat = routes.ravel()
    valid = flat[(flat >= 0) & (flat < n)]
    return np.bincount(valid, minlength=n)


def load_stats(loads: np.ndarray) -> dict:
    """Compute load distribution statistics."""
    return {
        "mean": float(np.mean(loads)),
        "std": float(np.std(loads)),
        "max": int(np.max(loads)),
        "min": int(np.min(loads)),
        "cv": float(np.std(loads) / np.mean(loads)) if np.mean(loads) > 0 else 0.0,
        "max_over_mean": float(np.max(loads) / np.mean(loads)) if np.mean(loads) > 0 else 0.0,
    }


def survival_rate(routes: np.ndarray, n: int) -> float:
    """Fraction of route slots that have a valid assignment."""
    flat = routes.ravel()
    return float(np.sum((flat >= 0) & (flat < n)) / len(flat))


# ── High-level API ───────────────────────────────────────────────────────

def route(
    source_weights=None,
    target_capacities=None,
    n: int = 1024,
    k: int = 4,
    seed: int = 42,
    density: float = 0.3,
) -> np.ndarray:
    """
    Route n sources to n targets using the Phase Router.

    Parameters:
        source_weights:     Per-source demand (list/array of length n).
                            If None, all sources have equal weight.
        target_capacities:  Relative capacity per target (list/array of length n).
                            If None, all targets have equal capacity.
        n:                  Number of sources = number of targets.
        k:                  Fan-out (max targets per source).
        seed:               Deterministic seed.
        density:            Base density of the bit matrices (fraction of n).

    Returns:
        np.ndarray of shape (n, k), dtype int32.
        routes[i] contains up to k target indices for source i (-1 = empty).
    """
    if _rs is None:
        raise ImportError(
            "phase_router_rs not installed. Run: maturin develop --release"
        )

    # Default: uniform weights / capacities
    if source_weights is None:
        source_weights = np.ones(n, dtype=np.float64)
    else:
        source_weights = np.asarray(source_weights, dtype=np.float64)
        n = len(source_weights)

    if target_capacities is None:
        target_capacities = np.ones(n, dtype=np.float64)
    else:
        target_capacities = np.asarray(target_capacities, dtype=np.float64)

    assert len(source_weights) == n
    assert len(target_capacities) == n

    nb_words = (n + 63) // 64

    # Source bit matrix: ones proportional to weight
    s_mean = np.mean(source_weights)
    s_ones = np.round((source_weights / s_mean) * density * n).clip(1, n).astype(int)
    s_bits = build_bits(s_ones, n)

    # Target bit matrix: ones proportional to capacity
    t_mean = np.mean(target_capacities)
    t_ones = np.round((target_capacities / t_mean) * density * n).clip(1, n).astype(int)
    t_bits = build_bits(t_ones, n)

    # Ensure contiguous
    s_bits = np.ascontiguousarray(s_bits, dtype=np.uint64)
    t_bits = np.ascontiguousarray(t_bits, dtype=np.uint64)

    # Call Rust kernel (auto-generates permutations from seed)
    routes = _rs.phase_router_auto(s_bits, t_bits, n, k, seed)

    return routes


# ── Capacity profiles ────────────────────────────────────────────────────

def uniform_capacities(n: int) -> np.ndarray:
    """All experts have equal capacity."""
    return np.ones(n, dtype=np.float64)


def strong_hetero_capacities(n: int, seed: int = 0) -> np.ndarray:
    """10% at 8×, 20% at 2×, rest at 1×."""
    rng = np.random.default_rng(seed)
    caps = np.ones(n, dtype=np.float64)
    t1 = int(n * 0.1)
    t2 = int(n * 0.2)
    caps[:t1] = 8.0
    caps[t1:t1 + t2] = 2.0
    rng.shuffle(caps)
    return caps


def extreme_hetero_capacities(n: int, seed: int = 0) -> np.ndarray:
    """5% at 16×, rest at 1×."""
    rng = np.random.default_rng(seed)
    caps = np.ones(n, dtype=np.float64)
    top = max(int(n * 0.05), 1)
    caps[:top] = 16.0
    rng.shuffle(caps)
    return caps


def make_hard_caps(
    rel_caps: np.ndarray, n: int, k: int, headroom: float = 1.2
) -> np.ndarray:
    """Convert relative capacities to hard integer caps with headroom."""
    total = rel_caps.sum()
    hard = np.ceil((rel_caps / total) * n * k * headroom).clip(1).astype(int)
    return hard

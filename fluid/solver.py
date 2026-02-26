"""
solver.py — Pressure Projection (Member 2's Domain)
=====================================================
This file is Squad A Member 2's responsibility, but it lives here
so the physics loop is complete and testable without waiting.

The pressure projection step enforces INCOMPRESSIBILITY:
  div(v) = 0 everywhere

After advection, the velocity field is generally NOT divergence-free
(fluid "piles up" in some cells). We fix this by:
  1. Computing divergence of the current velocity field
  2. Solving the Poisson equation for pressure: ∇²p = div(v)
  3. Subtracting the pressure gradient from velocity: v = v - ∇p

This is called "Helmholtz-Hodge decomposition" — any vector field
can be decomposed into a divergence-free part + a curl-free part (gradient).
We want the divergence-free part.

The "Hybrid Switch" lives here: mode="PHYSICS" uses Jacobi iteration,
mode="ML" will call the Squad B model instead.
"""

import numpy as np
import time
from .grid import FluidGrid


# ── Hybrid mode switch ────────────────────────────────────────────────────────
# Member 2 owns this. Member 4 (ML) will plug into the "ML" branch.
MODE_PHYSICS = "PHYSICS"
MODE_ML      = "ML"


def project(grid: FluidGrid, iterations: int = 40, mode: str = MODE_PHYSICS,
            ml_model=None) -> dict:
    """
    Pressure projection: make the velocity field divergence-free.

    This is the most expensive step in the simulation (~80% of CPU time).
    The ML model will eventually replace this with fast inference.

    Args:
        grid       : The FluidGrid to modify in-place
        iterations : Jacobi iterations (more = more accurate, slower)
        mode       : "PHYSICS" = classical Jacobi, "ML" = Squad B's model
        ml_model   : ONNX/PyTorch model (only used when mode="ML")

    Returns:
        dict with timing and error metrics (for benchmarking)
    """
    t_start = time.perf_counter()

    if mode == MODE_PHYSICS:
        pressure, divergence = _project_jacobi(grid, iterations)
    elif mode == MODE_ML:
        if ml_model is None:
            raise ValueError("ML mode selected but no model provided. Pass ml_model=<your_onnx_session>")
        pressure, divergence = _project_ml(grid, ml_model)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'PHYSICS' or 'ML'.")

    # Store pressure in grid
    grid.pressure = pressure

    # Subtract pressure gradient from velocity (the "projection" step)
    _subtract_pressure_gradient(grid)

    # Re-apply boundary conditions
    grid.set_boundary()

    t_end = time.perf_counter()

    # Compute post-projection divergence for benchmarking
    div_after = grid.compute_divergence()

    return {
        "mode"          : mode,
        "time_ms"       : (t_end - t_start) * 1000,
        "iterations"    : iterations,
        "divergence_before_max" : np.abs(divergence).max(),
        "divergence_after_max"  : np.abs(div_after).max(),
        "divergence_after_mean" : np.abs(div_after).mean(),
    }


def _project_jacobi(grid: FluidGrid, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Classical Jacobi iterative pressure solver.

    Solves: ∇²p = div(v)  (Poisson equation)
    
    Jacobi iteration:
      p[i,j,k] = (1/6) * (sum of 6 neighbors) - (1/6) * div[i,j,k]
    
    This is the "Physics Baseline" — ground truth that the ML model
    must learn to approximate. Member 3 saves (velocity_input, pressure_output)
    pairs from this function as training data.

    Returns: (pressure field (N,N,N), divergence field (N,N,N))
    """
    N = grid.N
    dx = 1.0 / N

    # Step 1: Compute divergence of current velocity field
    divergence = grid.compute_divergence()

    # Step 2: Solve Poisson equation via Jacobi iteration
    p = np.zeros((N, N, N), dtype=np.float32)

    for _ in range(iterations):
        # Sum of 6 neighbors (interior only, via slicing — no Python loops!)
        p_neighbors = (
            p[2:,  1:-1, 1:-1] +
            p[:-2, 1:-1, 1:-1] +
            p[1:-1, 2:,  1:-1] +
            p[1:-1, :-2, 1:-1] +
            p[1:-1, 1:-1, 2: ] +
            p[1:-1, 1:-1, :-2]
        )

        # Jacobi update
        p[1:-1, 1:-1, 1:-1] = (p_neighbors - dx * dx * divergence[1:-1, 1:-1, 1:-1]) / 6.0

        # Neumann BCs: dp/dn = 0 at walls
        p[0,  :, :] = p[1,  :, :]
        p[-1, :, :] = p[-2, :, :]
        p[:, 0,  :] = p[:, 1,  :]
        p[:, -1, :] = p[:, -2, :]
        p[:, :,  0] = p[:, :,  1]
        p[:, :, -1] = p[:, :, -2]

    return p, divergence


def _project_ml(grid: FluidGrid, ml_model) -> tuple[np.ndarray, np.ndarray]:
    """
    ML-based pressure prediction (Squad B's ONNX model).
    
    This replaces the slow Jacobi loop with a single neural network inference.
    Expected speedup: 5–20x depending on model size.

    Input to model:  velocity divergence field (N,N,N) 
    Output of model: predicted pressure field (N,N,N)

    Member 4 is responsible for making this interface work.
    """
    divergence = grid.compute_divergence()

    # Prepare input tensor: shape (1, 1, N, N, N) — batch + channel dims
    input_tensor = divergence[np.newaxis, np.newaxis, :, :, :]

    # Run ONNX inference
    # Member 4: make sure your model's input name matches 'divergence_input'
    output = ml_model.run(
        output_names=None,
        input_feed={"divergence_input": input_tensor.astype(np.float32)}
    )

    pressure = output[0][0, 0, :, :, :]  # Remove batch + channel dims
    return pressure.astype(np.float32), divergence


def _subtract_pressure_gradient(grid: FluidGrid):
    """
    Subtract ∇p from velocity to make it divergence-free.

    v_new = v_old - ∇p

    The gradient is computed using central differences at cell faces
    (since velocity lives on faces in the staggered MAC grid).

    This is what actually "fixes" the velocity field — the Jacobi
    solver above just finds the pressure; this step applies it.
    """
    N = grid.N
    dx = 1.0 / N

    # du/dx at interior X-faces: p[i] - p[i-1]
    grid.u[1:-1, :, :] -= (grid.pressure[1:, :, :] - grid.pressure[:-1, :, :]) / dx

    # dv/dy at interior Y-faces
    grid.v[:, 1:-1, :] -= (grid.pressure[:, 1:, :] - grid.pressure[:, :-1, :]) / dx

    # dw/dz at interior Z-faces
    grid.w[:, :, 1:-1] -= (grid.pressure[:, :, 1:] - grid.pressure[:, :, :-1]) / dx

"""
diffuse.py — Diffusion via Jacobi Iteration
============================================
Diffusion makes fluids spread out over time.
  - High diffusion  → density spreads fast (watercolor bleed)
  - Low diffusion   → density stays tight (laser-focused smoke column)
  - High viscosity  → thick fluid (honey)
  - Low viscosity   → thin fluid (air, water)

The math: We need to solve the implicit heat equation:
  (I - α·∇²) x_new = x_old

where α = diffusion_rate * dt * N²

Why implicit? Because explicit diffusion (just adding the Laplacian each step)
is only stable when dt is tiny. Implicit diffusion is unconditionally stable —
you can use large dt and the simulation won't blow up.

Solving this exactly is expensive. Instead we use Jacobi Iteration:
a simple loop where each cell averages its 6 neighbors repeatedly.
After ~20 iterations it converges to a good-enough solution.

Gauss-Seidel (updating in-place) converges faster but harder to vectorize.
Jacobi (ping-pong between two buffers) is slower but fully NumPy-vectorized.
We use Jacobi.
"""

import numpy as np
from .grid import FluidGrid


def _jacobi_solve(
    field: np.ndarray,
    field_prev: np.ndarray,
    alpha: float,
    beta: float,
    iterations: int = 20
) -> np.ndarray:
    """
    Jacobi iteration solver for: (I - alpha * Laplacian) * x = b
    Rearranged to: x_new[i,j,k] = (b[i,j,k] + alpha * sum_of_6_neighbors) / beta

    This is fully vectorized — no Python loops over cells.
    We use array slicing: field[1:-1, 1:-1, 1:-1] to operate on the interior.

    Args:
        field      : Current field (will be iteratively refined), shape (Nx, Ny, Nz)
        field_prev : Right-hand side (original values), same shape
        alpha      : Diffusion coefficient (dt * rate * N²)
        beta       : Normalization (6 * alpha + 1 for density, or as specified)
        iterations : Number of Jacobi sweeps (20 is usually enough)

    Returns:
        Solved field (same shape as input)
    """
    x = field.copy()
    b = field_prev

    for _ in range(iterations):
        # Sum of 6 face neighbors (interior cells only, via slicing)
        # Shape: (Nx-2, Ny-2, Nz-2) — we skip the boundary layer
        neighbors = (
            x[2:,  1:-1, 1:-1] +   # x+1 neighbor
            x[:-2, 1:-1, 1:-1] +   # x-1 neighbor
            x[1:-1, 2:,  1:-1] +   # y+1 neighbor
            x[1:-1, :-2, 1:-1] +   # y-1 neighbor
            x[1:-1, 1:-1, 2: ] +   # z+1 neighbor
            x[1:-1, 1:-1, :-2]     # z-1 neighbor
        )

        # Jacobi update: new value = (source + alpha * neighbors) / beta
        x[1:-1, 1:-1, 1:-1] = (b[1:-1, 1:-1, 1:-1] + alpha * neighbors) / beta

        # Boundary conditions are applied after each iteration
        # (simple Neumann: copy boundary from nearest interior)
        x[0,  :, :] = x[1,  :, :]
        x[-1, :, :] = x[-2, :, :]
        x[:, 0,  :] = x[:, 1,  :]
        x[:, -1, :] = x[:, -2, :]
        x[:, :,  0] = x[:, :,  1]
        x[:, :, -1] = x[:, :, -2]

    return x


def diffuse_density(grid: FluidGrid):
    """
    Apply diffusion to the density field.
    
    This spreads smoke/density to neighboring cells based on the diffusion rate.
    Uses an implicit solve so it's stable even with large dt.

    Modifies: grid.density (in-place)
    """
    if grid.diffusion == 0.0:
        return  # Skip if no diffusion (saves time)

    N = grid.N
    # alpha: how much diffusion happens this timestep
    # N² factor accounts for the fact that we work in grid-index space
    alpha = grid.dt * grid.diffusion * N * N
    beta  = 1.0 + 6.0 * alpha

    grid.density = _jacobi_solve(
        field=grid.density,
        field_prev=grid.density_prev,
        alpha=alpha,
        beta=beta,
        iterations=20
    )


def diffuse_velocity(grid: FluidGrid):
    """
    Apply viscous diffusion to all three velocity components.
    
    Makes the velocity field "spread out" — high viscosity means the fluid
    resists shearing (honey), low viscosity means it flows freely (air).

    Each velocity component (u, v, w) is diffused independently since
    they live on different staggered grids.

    Modifies: grid.u, grid.v, grid.w (in-place)
    """
    if grid.viscosity == 0.0:
        return  # Skip for inviscid fluids

    N = grid.N
    alpha = grid.dt * grid.viscosity * N * N
    beta  = 1.0 + 6.0 * alpha

    grid.u = _jacobi_solve(grid.u, grid.u_prev, alpha, beta, iterations=20)
    grid.v = _jacobi_solve(grid.v, grid.v_prev, alpha, beta, iterations=20)
    grid.w = _jacobi_solve(grid.w, grid.w_prev, alpha, beta, iterations=20)

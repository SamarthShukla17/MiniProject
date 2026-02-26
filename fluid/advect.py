"""
advect.py — Semi-Lagrangian Advection
======================================
This is what makes fluid look like it's *actually flowing*.

The algorithm (per cell):
  1. Look at the current cell center position.
  2. Trace BACKWARD along the velocity field by one timestep (dt).
     → "Where did the stuff in this cell come FROM?"
  3. Sample the density/velocity at that back-traced position
     using trilinear interpolation (it'll land between grid cells).
  4. That sampled value becomes the new value for this cell.

Why "Semi-Lagrangian"?
  - Pure Lagrangian: track particles → complex, needs re-meshing
  - Pure Eulerian: look at fixed grid cells → numerically unstable
  - Semi-Lagrangian: use a fixed grid (Eulerian) BUT trace particles
    backward (Lagrangian thinking). Unconditionally stable. ✓

Key reference: Jos Stam, "Stable Fluids" (SIGGRAPH 1999)
"""

import numpy as np
from .grid import FluidGrid


def _trilinear_interpolate(field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Trilinear interpolation of a 3D scalar field at arbitrary positions.

    Given a field of shape (Nx, Ny, Nz) and arrays of query positions
    x ∈ [0, Nx-1], y ∈ [0, Ny-1], z ∈ [0, Nz-1], returns interpolated values.

    Trilinear interpolation = linear interp in X, then Y, then Z.
    Think of it as a weighted average of the 8 surrounding corner values.

    Args:
        field : 3D numpy array to sample from
        x, y, z : Query positions (same shape, can be fractional)

    Returns:
        Interpolated values, same shape as x/y/z
    """
    Nx, Ny, Nz = field.shape

    # Clamp positions to valid range (handles boundaries)
    x = np.clip(x, 0, Nx - 1.001)
    y = np.clip(y, 0, Ny - 1.001)
    z = np.clip(z, 0, Nz - 1.001)

    # Integer part (lower corner of the 8-cell cube)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    z0 = np.floor(z).astype(np.int32)

    # Upper corner
    x1 = np.minimum(x0 + 1, Nx - 1)
    y1 = np.minimum(y0 + 1, Ny - 1)
    z1 = np.minimum(z0 + 1, Nz - 1)

    # Fractional part (how far we are between lower and upper)
    tx = x - x0
    ty = y - y0
    tz = z - z0

    # Sample all 8 corners of the surrounding cube
    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]

    # Trilinear blend: lerp in X, then Y, then Z
    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    return c0 * (1 - tz) + c1 * tz


def advect_density(grid: FluidGrid):
    """
    Advect the density (smoke) field through the velocity field.

    For each cell center, we trace backward in time by dt, sample
    the density at the back-traced position, and store it as the new density.

    Modifies: grid.density (in-place, via temp buffer)
    """
    N = grid.N
    dt = grid.dt

    # Get cell-centered velocities (interpolated from staggered faces)
    uc, vc, wc = grid.get_velocity_at_center()

    # Build arrays of cell center positions: indices (i, j, k) for all cells
    i, j, k = np.meshgrid(
        np.arange(N, dtype=np.float32),
        np.arange(N, dtype=np.float32),
        np.arange(N, dtype=np.float32),
        indexing='ij'
    )

    # Back-trace: where did the fluid in cell (i,j,k) come FROM?
    # pos_back = current_pos - velocity * dt
    # Multiply dt by N to convert from world-space to grid-index-space
    x_back = i - dt * N * uc
    y_back = j - dt * N * vc
    z_back = k - dt * N * wc

    # Sample density at back-traced position
    grid.density = _trilinear_interpolate(grid.density_prev, x_back, y_back, z_back)


def advect_velocity(grid: FluidGrid):
    """
    Advect the velocity field through itself (self-advection).

    Each velocity component lives on its own staggered face, so we
    need to handle each component (u, v, w) separately, tracing back
    from the face-center positions.

    Modifies: grid.u, grid.v, grid.w (in-place, via temp buffers)
    """
    N = grid.N
    dt = grid.dt

    # ── Advect u (X-faces, shape N+1 × N × N) ────────────────────────────
    # u-face positions: (i+0.5, j, k) in grid index space → i ∈ [0..N]
    iu, ju, ku = np.meshgrid(
        np.arange(N + 1, dtype=np.float32),
        np.arange(N,     dtype=np.float32),
        np.arange(N,     dtype=np.float32),
        indexing='ij'
    )

    # Sample velocity at u-face positions by interpolating from cell centers
    # (simple: u at face is just u itself; for cross-terms, use cell-center interp)
    uc, vc, wc = grid.get_velocity_at_center()

    # For u-faces, interpolate vc and wc to u-face positions
    # u at face i ≈ average of cell i-1 and cell i
    def interp_to_u_faces(field_cc):
        """Interpolate a cell-center field (N,N,N) to u-face positions (N+1,N,N)."""
        padded = np.pad(field_cc, ((1, 1), (0, 0), (0, 0)), mode='edge')
        return 0.5 * (padded[:-1, :, :] + padded[1:, :, :])

    def interp_to_v_faces(field_cc):
        """Interpolate a cell-center field to v-face positions (N,N+1,N)."""
        padded = np.pad(field_cc, ((0, 0), (1, 1), (0, 0)), mode='edge')
        return 0.5 * (padded[:, :-1, :] + padded[:, 1:, :])

    def interp_to_w_faces(field_cc):
        """Interpolate a cell-center field to w-face positions (N,N,N+1)."""
        padded = np.pad(field_cc, ((0, 0), (0, 0), (1, 1)), mode='edge')
        return 0.5 * (padded[:, :, :-1] + padded[:, :, 1:])

    vc_at_u = interp_to_u_faces(vc)
    wc_at_u = interp_to_u_faces(wc)

    x_back_u = iu - dt * N * grid.u_prev
    y_back_u = ju - dt * N * vc_at_u
    z_back_u = ku - dt * N * wc_at_u
    grid.u = _trilinear_interpolate(grid.u_prev, x_back_u, y_back_u, z_back_u)

    # ── Advect v (Y-faces, shape N × N+1 × N) ────────────────────────────
    iv, jv, kv = np.meshgrid(
        np.arange(N,     dtype=np.float32),
        np.arange(N + 1, dtype=np.float32),
        np.arange(N,     dtype=np.float32),
        indexing='ij'
    )

    uc_at_v = interp_to_v_faces(uc)
    wc_at_v = interp_to_v_faces(wc)

    x_back_v = iv - dt * N * uc_at_v
    y_back_v = jv - dt * N * grid.v_prev
    z_back_v = kv - dt * N * wc_at_v
    grid.v = _trilinear_interpolate(grid.v_prev, x_back_v, y_back_v, z_back_v)

    # ── Advect w (Z-faces, shape N × N × N+1) ────────────────────────────
    iw, jw, kw = np.meshgrid(
        np.arange(N,     dtype=np.float32),
        np.arange(N,     dtype=np.float32),
        np.arange(N + 1, dtype=np.float32),
        indexing='ij'
    )

    uc_at_w = interp_to_w_faces(uc)
    vc_at_w = interp_to_w_faces(vc)

    x_back_w = iw - dt * N * uc_at_w
    y_back_w = jw - dt * N * vc_at_w
    z_back_w = kw - dt * N * grid.w_prev
    grid.w = _trilinear_interpolate(grid.w_prev, x_back_w, y_back_w, z_back_w)

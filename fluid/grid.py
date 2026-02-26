"""
grid.py — MAC (Marker-and-Cell) Staggered Grid
================================================
The foundation of the entire simulation.

Layout on a single cell:
  - Pressure `p` and Density `d` live at CELL CENTERS  → shape (N, N, N)
  - Velocity `u` lives on X-FACES                       → shape (N+1, N, N)
  - Velocity `v` lives on Y-FACES                       → shape (N, N+1, N)
  - Velocity `w` lives on Z-FACES                       → shape (N, N, N+1)

Why staggered? It prevents the "checkerboard" pressure instability
that appears on collocated grids. Industry standard since the 1960s.
"""

import numpy as np


class FluidGrid:
    """
    32³ MAC Grid storing all simulation state.
    This is the single source of truth passed between all physics steps.
    """

    def __init__(self, N: int = 32, dt: float = 0.1, diffusion: float = 0.0001, viscosity: float = 0.0001):
        """
        Args:
            N          : Grid resolution (32 means 32³ cells)
            dt         : Timestep in seconds
            diffusion  : How fast density/smoke spreads (0 = no spreading)
            viscosity  : Fluid thickness (0 = inviscid like air, high = honey)
        """
        self.N = N
        self.dt = dt
        self.diffusion = diffusion
        self.viscosity = viscosity

        # ── Scalar fields (cell-centered) ──────────────────────────────────
        self.density     = np.zeros((N, N, N), dtype=np.float32)
        self.density_prev = np.zeros((N, N, N), dtype=np.float32)
        self.pressure    = np.zeros((N, N, N), dtype=np.float32)

        # Temperature field (drives buoyancy — hot smoke rises)
        self.temperature = np.zeros((N, N, N), dtype=np.float32)

        # ── Velocity fields (face-centered, staggered) ─────────────────────
        # u: X-component, lives on X-faces → (N+1, N, N)
        # v: Y-component, lives on Y-faces → (N, N+1, N)
        # w: Z-component, lives on Z-faces → (N, N, N+1)
        self.u = np.zeros((N + 1, N, N), dtype=np.float32)
        self.v = np.zeros((N, N + 1, N), dtype=np.float32)
        self.w = np.zeros((N, N, N + 1), dtype=np.float32)

        # Previous velocity (needed for advection)
        self.u_prev = np.zeros_like(self.u)
        self.v_prev = np.zeros_like(self.v)
        self.w_prev = np.zeros_like(self.w)

    def add_density(self, x: int, y: int, z: int, amount: float, radius: int = 2):
        """
        Inject density (smoke) into the grid at cell (x, y, z).
        Uses a small radius so the injection looks smooth, not a single pixel.

        Args:
            x, y, z : Cell indices (integers, 0 to N-1)
            amount  : How much density to add
            radius  : Injection radius in cells
        """
        N = self.N
        x0, x1 = max(0, x - radius), min(N, x + radius + 1)
        y0, y1 = max(0, y - radius), min(N, y + radius + 1)
        z0, z1 = max(0, z - radius), min(N, z + radius + 1)
        self.density[x0:x1, y0:y1, z0:z1] += amount
        self.density_prev[x0:x1, y0:y1, z0:z1] += amount

    def add_velocity(self, x: int, y: int, z: int, du: float, dv: float, dw: float):
        """
        Apply a velocity impulse at cell (x, y, z).
        Because of staggering, we add to the nearest face indices.

        Args:
            x, y, z     : Cell indices
            du, dv, dw  : Velocity components to add (m/s)
        """
        self.u[x, y, z]     += du
        self.u[x + 1, y, z] += du
        self.v[x, y, z]     += dv
        self.v[x, y + 1, z] += dv
        self.w[x, y, z]     += dw
        self.w[x, y, z + 1] += dw

    def set_boundary(self):
        """
        Enforce no-slip, no-penetration boundary conditions on all 6 walls.
        
        - Normal velocity at walls = 0 (fluid can't pass through the box)
        - Tangential velocity at walls = negated from interior (no-slip)
        - Density/pressure are extrapolated (Neumann condition)
        
        Without this, fluid leaks out of the array → simulation explodes.
        """
        N = self.N

        # ── u (X-velocity): zero normal component at X walls ──────────────
        self.u[0,  :, :] = 0.0
        self.u[N,  :, :] = 0.0
        # No-slip: mirror interior
        self.u[1,  :, :] = -self.u[2,  :, :]
        self.u[N-1,:, :] = -self.u[N-2,:, :]

        # ── v (Y-velocity): zero normal component at Y walls ──────────────
        self.v[:, 0,  :] = 0.0
        self.v[:, N,  :] = 0.0
        self.v[:, 1,  :] = -self.v[:, 2,  :]
        self.v[:, N-1,:] = -self.v[:, N-2,:]

        # ── w (Z-velocity): zero normal component at Z walls ──────────────
        self.w[:, :, 0 ] = 0.0
        self.w[:, :, N ] = 0.0
        self.w[:, :, 1 ] = -self.w[:, :, 2 ]
        self.w[:, :, N-1] = -self.w[:, :, N-2]

        # ── Density: copy nearest interior cell (Neumann BC) ──────────────
        self.density[0,  :, :] = self.density[1,  :, :]
        self.density[N-1,:, :] = self.density[N-2,:, :]
        self.density[:, 0,  :] = self.density[:, 1,  :]
        self.density[:,N-1, :] = self.density[:, N-2,:]
        self.density[:, :, 0 ] = self.density[:, :, 1 ]
        self.density[:, :,N-1] = self.density[:, :,N-2]

    def save_state(self) -> dict:
        """
        Snapshot current state as numpy arrays.
        Called by Member 3 (Data Pipeline) to build the ML training dataset.

        Returns a dict with:
          'velocity_u', 'velocity_v', 'velocity_w' → input features for ML
          'density', 'pressure'                    → labels / targets
        """
        return {
            "velocity_u": self.u.copy(),
            "velocity_v": self.v.copy(),
            "velocity_w": self.w.copy(),
            "density":    self.density.copy(),
            "pressure":   self.pressure.copy(),
        }

    def get_velocity_at_center(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate staggered face velocities to cell centers.
        Useful for visualization (Member 5) and divergence calculation (Member 2).

        Returns (uc, vc, wc) each of shape (N, N, N).
        """
        uc = 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])
        vc = 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])
        wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return uc, vc, wc

    def compute_divergence(self) -> np.ndarray:
        """
        Compute divergence of the velocity field.
        div(v) = du/dx + dv/dy + dw/dz

        For an incompressible fluid, this should be ~0 everywhere.
        Member 2 will use this to verify their pressure solver.
        High divergence = broken simulation.

        Returns: (N, N, N) array of divergence values.
        """
        dx = 1.0 / self.N
        div = (
            (self.u[1:, :, :] - self.u[:-1, :, :]) / dx +
            (self.v[:, 1:, :] - self.v[:, :-1, :]) / dx +
            (self.w[:, :, 1:] - self.w[:, :, :-1]) / dx
        )
        return div

    def reset(self):
        """Zero out all fields. Useful for running multiple simulations."""
        for arr in [self.density, self.density_prev, self.pressure,
                    self.temperature, self.u, self.v, self.w,
                    self.u_prev, self.v_prev, self.w_prev]:
            arr[:] = 0.0

    def __repr__(self):
        max_div = np.abs(self.compute_divergence()).max()
        max_vel = max(np.abs(self.u).max(), np.abs(self.v).max(), np.abs(self.w).max())
        return (
            f"FluidGrid(N={self.N}, dt={self.dt})\n"
            f"  density  : max={self.density.max():.4f}, sum={self.density.sum():.2f}\n"
            f"  velocity : max_magnitude={max_vel:.4f}\n"
            f"  divergence: max={max_div:.6f} (target: ~0)"
        )

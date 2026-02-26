"""
simulation.py — Master Physics Loop (Squad A Handoff)
======================================================
This is the complete simulation step that ties everything together.
One call to `step()` advances the fluid by dt seconds.

Physics pipeline per frame:
  1. Apply external forces (buoyancy, wind, user input)
  2. Diffuse velocity (viscosity)
  3. Project velocity (enforce incompressibility)  ← Member 2's step
  4. Advect velocity (self-advection)
  5. Project again (clean up after advection)
  6. Diffuse density (smoke spreading)
  7. Advect density (smoke movement)
  8. Apply boundary conditions

This follows the "Stable Fluids" paper by Jos Stam exactly.
"""

import numpy as np
import time
from .grid import FluidGrid
from .advect import advect_density, advect_velocity
from .diffuse import diffuse_density, diffuse_velocity
from .forces import apply_buoyancy, heat_source
from .solver import project, MODE_PHYSICS


class FluidSimulation:
    """
    The complete 3D fluid simulation.
    
    Usage:
        sim = FluidSimulation(N=32)
        sim.add_smoke_source(16, 4, 16)   # Inject smoke at bottom-center
        for frame in range(100):
            sim.step()
            density = sim.grid.density     # Hand to visualizer
    """

    def __init__(self, N: int = 32, dt: float = 0.1,
                 diffusion: float = 0.00005, viscosity: float = 0.00001):
        """
        Args:
            N          : Grid resolution (32 → 32³ = 32,768 cells)
            dt         : Timestep (seconds). 0.1 = 10 FPS physics update
            diffusion  : Smoke spreading rate (keep small for clean smoke)
            viscosity  : Fluid thickness (keep very small for air-like smoke)
        """
        self.grid = FluidGrid(N=N, dt=dt, diffusion=diffusion, viscosity=viscosity)
        self.frame = 0
        self.mode = MODE_PHYSICS
        self.ml_model = None
        self.perf_log = []   # stores timing data per frame

    def set_mode(self, mode: str, ml_model=None):
        """
        Switch between Physics and ML mode.
        Called by Member 6 (System Integrator) for the live toggle.

        Args:
            mode     : "PHYSICS" or "ML"
            ml_model : ONNX runtime session (required if mode="ML")
        """
        self.mode = mode
        self.ml_model = ml_model
        print(f"[Simulation] Mode switched to: {mode}")

    def add_smoke_source(self, x: int, y: int, z: int,
                         density_rate: float = 5.0,
                         velocity: tuple = (0.0, 2.0, 0.0),
                         temperature: float = 1.0):
        """
        Define a continuous smoke emitter.
        Call this before stepping to inject smoke + heat each frame.

        Args:
            x, y, z      : Source position (cell indices)
            density_rate : How much density to add per frame
            velocity     : (du, dv, dw) — upward velocity by default
            temperature  : Heat to inject (drives buoyancy)
        """
        self.grid.add_density(x, y, z, density_rate)
        self.grid.add_velocity(x, y, z, *velocity)
        heat_source(self.grid, x, y, z, temperature)

    def step(self, solver_iterations: int = 40) -> dict:
        """
        Advance simulation by one timestep (dt seconds).
        
        Returns performance metrics dict for benchmarking.

        Args:
            solver_iterations : Jacobi iterations for pressure solver.
                                Higher = more accurate, slower.
                                20 = fast/rough, 40 = balanced, 80 = accurate
        """
        t_total_start = time.perf_counter()
        g = self.grid

        # ── Save previous state for advection back-tracing ─────────────────
        np.copyto(g.u_prev, g.u)
        np.copyto(g.v_prev, g.v)
        np.copyto(g.w_prev, g.w)
        np.copyto(g.density_prev, g.density)

        # ── Step 1: External forces ────────────────────────────────────────
        t0 = time.perf_counter()
        apply_buoyancy(g)
        t_forces = (time.perf_counter() - t0) * 1000

        # ── Step 2: Diffuse velocity (viscosity) ───────────────────────────
        t0 = time.perf_counter()
        diffuse_velocity(g)
        g.set_boundary()
        t_diffuse_vel = (time.perf_counter() - t0) * 1000

        # ── Step 3: Project velocity (enforce incompressibility) ───────────
        # This is the expensive step. Mode switch lives here.
        t0 = time.perf_counter()
        proj_metrics = project(g, iterations=solver_iterations,
                               mode=self.mode, ml_model=self.ml_model)
        t_project1 = (time.perf_counter() - t0) * 1000

        # ── Step 4: Advect velocity (self-advection) ───────────────────────
        t0 = time.perf_counter()
        # Update prev again after projection (cleaner advection source)
        np.copyto(g.u_prev, g.u)
        np.copyto(g.v_prev, g.v)
        np.copyto(g.w_prev, g.w)
        advect_velocity(g)
        g.set_boundary()
        t_advect_vel = (time.perf_counter() - t0) * 1000

        # ── Step 5: Project again (clean up post-advection divergence) ─────
        t0 = time.perf_counter()
        project(g, iterations=solver_iterations,
                mode=self.mode, ml_model=self.ml_model)
        t_project2 = (time.perf_counter() - t0) * 1000

        # ── Step 6: Diffuse density (smoke spreading) ──────────────────────
        t0 = time.perf_counter()
        diffuse_density(g)
        t_diffuse_den = (time.perf_counter() - t0) * 1000

        # ── Step 7: Advect density (smoke movement) ────────────────────────
        t0 = time.perf_counter()
        np.copyto(g.density_prev, g.density)
        advect_density(g)
        g.set_boundary()
        t_advect_den = (time.perf_counter() - t0) * 1000

        # ── Frame bookkeeping ──────────────────────────────────────────────
        self.frame += 1
        t_total = (time.perf_counter() - t_total_start) * 1000

        metrics = {
            "frame"            : self.frame,
            "mode"             : self.mode,
            "total_ms"         : t_total,
            "fps"              : 1000.0 / t_total if t_total > 0 else 0,
            "forces_ms"        : t_forces,
            "diffuse_vel_ms"   : t_diffuse_vel,
            "project1_ms"      : t_project1,
            "advect_vel_ms"    : t_advect_vel,
            "project2_ms"      : t_project2,
            "diffuse_den_ms"   : t_diffuse_den,
            "advect_den_ms"    : t_advect_den,
            "divergence_max"   : proj_metrics["divergence_after_max"],
            "divergence_mean"  : proj_metrics["divergence_after_mean"],
            "density_total"    : float(g.density.sum()),
        }
        self.perf_log.append(metrics)
        return metrics

    def get_dataset_snapshot(self) -> dict:
        """
        Capture the current state as a training data point.
        Called by Member 3 (Data Pipeline Engineer) after each step.

        Format that Squad B (ML) expects:
          Input  → velocity divergence field (the "problem")
          Target → pressure field that fixes it (the "solution")

        Returns dict ready to be saved as .npy files.
        """
        return {
            "frame"         : self.frame,
            "velocity_u"    : self.grid.u.copy(),
            "velocity_v"    : self.grid.v.copy(),
            "velocity_w"    : self.grid.w.copy(),
            "divergence"    : self.grid.compute_divergence().copy(),  # ML input
            "pressure"      : self.grid.pressure.copy(),              # ML target
            "density"       : self.grid.density.copy(),
        }

    def print_status(self):
        """Pretty-print current simulation state."""
        g = self.grid
        div = g.compute_divergence()
        uc, vc, wc = g.get_velocity_at_center()
        print(f"\n{'='*50}")
        print(f"  Frame: {self.frame}  |  Mode: {self.mode}")
        print(f"  Density   : max={g.density.max():.4f}, total={g.density.sum():.2f}")
        print(f"  Velocity  : max_u={np.abs(uc).max():.4f}, max_v={np.abs(vc).max():.4f}")
        print(f"  Divergence: max={np.abs(div).max():.6f}, mean={np.abs(div).mean():.8f}")
        print(f"  Pressure  : max={g.pressure.max():.4f}, min={g.pressure.min():.4f}")
        if self.perf_log:
            last = self.perf_log[-1]
            print(f"  Perf      : {last['total_ms']:.1f}ms/frame ({last['fps']:.1f} FPS)")
        print(f"{'='*50}")

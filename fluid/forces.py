"""
forces.py — External Forces (Buoyancy, Wind, User Input)
=========================================================
Applies body forces to the velocity field each timestep.

Buoyancy is the most important one: hot smoke rises because warm air
is less dense than cool air. We model this with a simple approximation:

  F_buoyancy = -β * density + κ * (T - T_ambient)

where:
  β = density coefficient (density weighs the fluid down)
  κ = thermal coefficient (temperature pushes fluid up)

This is applied as an upward force on the Y-velocity component (v).
"""

import numpy as np
from .grid import FluidGrid


# ── Buoyancy parameters ───────────────────────────────────────────────────────
BUOYANCY_DENSITY_COEFF = 0.1    # β: how much density pulls DOWN
BUOYANCY_THERMAL_COEFF = 0.08   # κ: how much temperature pushes UP
AMBIENT_TEMPERATURE    = 0.0    # T_ambient: room temperature in normalized units


def apply_buoyancy(grid: FluidGrid):
    """
    Apply buoyancy force to the Y-velocity (upward direction).

    Buoyancy force at each cell: F = -β*d + κ*(T - T_amb)
      - Smoke/density makes fluid heavier → pushes DOWN (negative Y)
      - Heat makes fluid lighter → pushes UP (positive Y)

    The v-field lives on Y-faces (shape N × N+1 × N), so we average
    the cell-center force to the two adjacent faces.

    Modifies: grid.v (in-place)
    """
    # Cell-centered buoyancy force
    # Shape: (N, N, N)
    T_diff = grid.temperature - AMBIENT_TEMPERATURE
    force_cc = (
        -BUOYANCY_DENSITY_COEFF * grid.density +
         BUOYANCY_THERMAL_COEFF * T_diff
    )  # positive = upward

    # Apply to Y-velocity faces (interpolate from cell centers to Y-faces)
    # v-face between cell j and j+1 gets average of those two cells' forces
    # Interior faces: average of lower and upper cell
    grid.v[:, 1:-1, :] += 0.5 * grid.dt * (force_cc[:, :-1, :] + force_cc[:, 1:, :])
    # Boundary faces: just use the edge cell
    grid.v[:, 0,  :] += grid.dt * force_cc[:, 0,  :]
    grid.v[:, -1, :] += grid.dt * force_cc[:, -1, :]


def apply_wind(grid: FluidGrid, direction: tuple = (1.0, 0.0, 0.0), strength: float = 0.5):
    """
    Apply a constant wind force across the entire domain.
    Useful for testing — blow smoke in a consistent direction.

    Args:
        direction : (du, dv, dw) unit vector of wind direction
        strength  : Wind speed magnitude
    """
    du, dv, dw = direction
    grid.u += strength * du * grid.dt
    grid.v += strength * dv * grid.dt
    grid.w += strength * dw * grid.dt


def apply_impulse(grid: FluidGrid, x: int, y: int, z: int,
                  fx: float, fy: float, fz: float, radius: int = 3):
    """
    Apply a localized force impulse (e.g., a fan, explosion, user click).
    Force falls off with distance from the center point.

    Args:
        x, y, z : Center of the impulse (cell indices)
        fx, fy, fz : Force components
        radius : Influence radius in cells
    """
    N = grid.N
    ix, jx, kx = np.meshgrid(
        np.arange(N + 1), np.arange(N), np.arange(N), indexing='ij'
    )
    dist_u = np.sqrt((ix - x)**2 + (jx - y)**2 + (kx - z)**2)
    mask_u = dist_u < radius
    grid.u[mask_u] += fx * grid.dt * (1 - dist_u[mask_u] / radius)

    iy, jy, ky = np.meshgrid(
        np.arange(N), np.arange(N + 1), np.arange(N), indexing='ij'
    )
    dist_v = np.sqrt((iy - x)**2 + (jy - y)**2 + (ky - z)**2)
    mask_v = dist_v < radius
    grid.v[mask_v] += fy * grid.dt * (1 - dist_v[mask_v] / radius)

    iz, jz, kz = np.meshgrid(
        np.arange(N), np.arange(N), np.arange(N + 1), indexing='ij'
    )
    dist_w = np.sqrt((iz - x)**2 + (jz - y)**2 + (kz - z)**2)
    mask_w = dist_w < radius
    grid.w[mask_w] += fz * grid.dt * (1 - dist_w[mask_w] / radius)


def heat_source(grid: FluidGrid, x: int, y: int, z: int,
                amount: float = 1.0, radius: int = 2):
    """
    Inject heat at a point (simulates a flame or heat source).
    Temperature drives buoyancy upward force.

    Args:
        x, y, z : Heat source location
        amount  : Temperature to add
        radius  : Injection radius
    """
    N = grid.N
    x0, x1 = max(0, x - radius), min(N, x + radius + 1)
    y0, y1 = max(0, y - radius), min(N, y + radius + 1)
    z0, z1 = max(0, z - radius), min(N, z + radius + 1)
    grid.temperature[x0:x1, y0:y1, z0:z1] += amount
    # Temperature decays toward ambient over time
    grid.temperature *= 0.99

"""
fluid/ — Squad A Physics Package
==================================
Exports the main interfaces other squads will use.

Member 5 (Viz) imports: FluidSimulation
Member 3 (Data) imports: FluidSimulation → get_dataset_snapshot()
Member 6 (Integrator) imports: FluidSimulation → set_mode(), step()
"""

from .grid import FluidGrid
from .simulation import FluidSimulation
from .solver import MODE_PHYSICS, MODE_ML

__all__ = ["FluidGrid", "FluidSimulation", "MODE_PHYSICS", "MODE_ML"]

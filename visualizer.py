"""
visualizer.py — 3-Slice Density Viewer (Squad C's Domain)
==========================================================
Renders three orthogonal slices of the 3D density field:
  - XY plane (top-down view)   → grid[:, :, N//2]
  - XZ plane (front view)      → grid[:, N//2, :]
  - YZ plane (side view)       → grid[N//2, :, :]

Uses matplotlib FuncAnimation for real-time updates.
Member 5 will extend this into a full UI with mode toggle.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Custom smoke colormap: transparent black → orange → white
SMOKE_COLORS = ["#000000", "#1a0a00", "#ff6a00", "#ffffff"]
smoke_cmap = LinearSegmentedColormap.from_list("smoke", SMOKE_COLORS)


class FluidVisualizer:
    """
    Real-time 3-slice viewer of the fluid simulation.
    
    Usage (standalone):
        from fluid import FluidSimulation
        from visualizer import FluidVisualizer
        
        sim = FluidSimulation(N=32)
        sim.add_smoke_source(16, 4, 16)
        viz = FluidVisualizer(sim)
        viz.run()  # Opens live window
    """

    def __init__(self, simulation, slice_axis: int = None):
        """
        Args:
            simulation : FluidSimulation instance
            slice_axis : Which axis to slice at center. None = show all 3.
        """
        self.sim = simulation
        self.N = simulation.grid.N
        self.mid = self.N // 2

        self._setup_figure()

    def _setup_figure(self):
        """Initialize the matplotlib figure with 3 subplots."""
        self.fig, self.axes = plt.subplots(1, 3, figsize=(14, 5))
        self.fig.patch.set_facecolor('#0a0a0a')

        titles = ["XY (top-down)  z=mid", "XZ (front)     y=mid", "YZ (side)      x=mid"]
        self.imgs = []

        dummy = np.zeros((self.N, self.N))

        for ax, title in zip(self.axes, titles):
            ax.set_facecolor('#0a0a0a')
            ax.set_title(title, color='#aaaaaa', fontsize=9, fontfamily='monospace')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')

            img = ax.imshow(
                dummy, cmap=smoke_cmap,
                vmin=0, vmax=2.0,
                interpolation='bilinear',
                origin='lower',
                aspect='equal'
            )
            self.imgs.append(img)

        self.title_text = self.fig.suptitle(
            "Fluid Sim — Frame 0 | PHYSICS mode | 0.0 FPS",
            color='#cccccc', fontsize=10, fontfamily='monospace'
        )

        plt.tight_layout()

    def _get_slices(self) -> tuple:
        """Extract 3 orthogonal slices from the density field."""
        d = self.sim.grid.density
        m = self.mid
        return (
            d[:, :, m].T,   # XY: transpose so X is horizontal
            d[:, m, :].T,   # XZ
            d[m, :, :].T,   # YZ
        )

    def update(self, frame_num):
        """Called by FuncAnimation each frame. Steps sim and updates plots."""
        # Continuous smoke source at bottom center
        N = self.N
        self.sim.add_smoke_source(N // 2, 2, N // 2,
                                  density_rate=3.0,
                                  velocity=(0.0, 1.5, 0.0),
                                  temperature=0.8)

        # Step the physics
        metrics = self.sim.step(solver_iterations=20)

        # Update the three slice images
        slices = self._get_slices()
        for img, s in zip(self.imgs, slices):
            img.set_data(s)

        # Update title with live stats
        self.title_text.set_text(
            f"Fluid Sim — Frame {metrics['frame']} | {metrics['mode']} mode | "
            f"{metrics['fps']:.1f} FPS | "
            f"div_max={metrics['divergence_max']:.5f}"
        )

        return self.imgs + [self.title_text]

    def run(self, fps: int = 10, frames: int = 500):
        """
        Start the live animation window.

        Args:
            fps    : Target animation frame rate
            frames : Total frames to render (None = infinite)
        """
        interval_ms = 1000 // fps
        self.anim = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            interval=interval_ms,
            blit=True
        )
        plt.show()

    def save_gif(self, path: str = "fluid_sim.gif", fps: int = 10, frames: int = 100):
        """Save animation as a GIF (for reports and demos)."""
        print(f"Rendering {frames} frames to {path}...")
        self.anim = animation.FuncAnimation(
            self.fig, self.update, frames=frames, interval=100, blit=True
        )
        writer = animation.PillowWriter(fps=fps)
        self.anim.save(path, writer=writer)
        print(f"Saved: {path}")

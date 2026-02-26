# Team Guide — 3D Fluid Simulation Project

> Read your section. Clone the repo. Use the Claude prompt provided. Build your part.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repo Structure](#repo-structure)
- [Setup (Everyone Does This)](#setup-everyone-does-this)
- [Member 1 — Fluid Dynamics Lead](#member-1--fluid-dynamics-lead)
- [Member 2 — Solver Architect](#member-2--solver-architect)
- [Member 3 — Data Pipeline Engineer](#member-3--data-pipeline-engineer)
- [Member 4 — Model Architect](#member-4--model-architect)
- [Member 5 — Visualization Specialist](#member-5--visualization-specialist)
- [Member 6 — System Integrator](#member-6--system-integrator)
- [How Everything Connects](#how-everything-connects)
- [Timeline](#timeline)

---

## Project Overview

We are building a **3D smoke/fluid simulator** that runs on CPU.

The simulation uses real physics equations to model how smoke moves through a 32×32×32 grid. The twist: we train a neural network to **replace the slowest physics step** with fast ML inference — and prove it's nearly as accurate but 10x faster.

**Final demo:** A live viewer where you can toggle between "Classical Physics" and "ML Mode" in real time, and see the FPS jump.

---

## Repo Structure

```
fluid_sim/
├── main.py               ← Entry point. Run this.
├── visualizer.py         ← Member 5 owns this
├── data_pipeline.py      ← Member 3 owns this
├── requirements.txt
├── README.md
├── GUIDE.md              ← You are here
└── fluid/
    ├── __init__.py
    ├── grid.py           ← Member 1 owns this
    ├── advect.py         ← Member 1 owns this
    ├── diffuse.py        ← Member 1 owns this
    ├── forces.py         ← Member 1 owns this
    ├── simulation.py     ← Member 1 owns this
    └── solver.py         ← Member 2 owns this
```

---

## Setup (Everyone Does This)

```bash
# 1. Clone the repo
git clone <repo-url>
cd fluid_sim

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it (do this every time you open a new terminal)
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify it works
python main.py --mode headless --frames 20
```

You should see output like:
```
Headless simulation | N=32 | 20 frames
Frame 000 | 74.9ms (13.3 FPS) | div_max=99.6 | density=375.6
...
Average: 52ms/frame (19 FPS)
```

If that works, your environment is good.

> **Note for Linux/Ubuntu users:** If `pip install` gives an "externally managed environment" error, the venv approach above is correct. Always use `source .venv/bin/activate` before running anything.

---

---

## Member 1 — Fluid Dynamics Lead

**Files you own:** `fluid/grid.py`, `fluid/advect.py`, `fluid/diffuse.py`, `fluid/forces.py`, `fluid/simulation.py`

**Your job is already done.** The physics core is implemented and working. Your role from here is:
1. Keep the code stable as other members build on top of it
2. Debug any instability (smoke exploding or disappearing) over long runs
3. Be the physics expert your team asks questions to

### What your code does

- **`grid.py`** — The 32³ MAC staggered grid. Stores velocity (u, v, w) on cell faces and density/pressure at cell centers. This is the data structure everything else operates on.
- **`advect.py`** — Semi-Lagrangian advection. Moves smoke through the velocity field each frame using backward particle tracing + trilinear interpolation.
- **`diffuse.py`** — Jacobi iteration for diffusion. Makes smoke spread to neighboring cells over time (viscosity).
- **`forces.py`** — Buoyancy. Hot smoke rises. Applied as an upward force on the Y-velocity each frame.
- **`simulation.py`** — Master loop. Calls all the above steps in the correct order every frame.

### How to test your work

```bash
# Basic physics check
python main.py --mode headless --frames 100

# Detailed per-step timing (hand this output to Member 2)
python main.py --mode benchmark --frames 50
```

### Stability test (important for Week 3)

Run for 500 frames and watch that density doesn't go to zero or infinity:

```bash
python main.py --mode headless --frames 500
```

The `density_total` column should grow steadily (smoke keeps being injected) and never suddenly drop to 0 or spike to millions. If it does, there's a bug in advection or boundary conditions.

### Key concepts to understand

**MAC Grid (Staggered Grid):** Velocity components don't live at cell centers — `u` (x-velocity) lives on the left/right faces of each cell, `v` lives on top/bottom faces, `w` lives on front/back faces. Pressure and density live at centers. This prevents numerical instabilities.

**Semi-Lagrangian Advection:** Instead of asking "where does this cell's smoke go?", we ask "where did this cell's smoke come from?" We trace backward in time by `dt`, sample the density at that position (with interpolation), and use it as the new value. This is why it's stable even with large timesteps.

**Divergence:** Measures how much fluid is entering vs leaving each cell. For incompressible flow, this must be zero everywhere. `compute_divergence()` in `grid.py` is what Member 2 uses to verify their solver works.

### Your Claude AI prompt

Copy and paste this into Claude when you need help:

```
I am Member 1 (Fluid Dynamics Lead) on a 3D fluid simulation project.
We are building a 32³ MAC staggered grid fluid simulator in Python/NumPy
following Jos Stam's "Stable Fluids" algorithm.

My files are: fluid/grid.py, fluid/advect.py, fluid/diffuse.py, fluid/forces.py, fluid/simulation.py

The grid structure:
- Pressure and density: shape (32, 32, 32) — cell centers
- u velocity (X): shape (33, 32, 32) — X faces
- v velocity (Y): shape (32, 33, 32) — Y faces
- w velocity (Z): shape (32, 32, 33) — Z faces

Physics pipeline per frame:
1. Apply buoyancy forces
2. Diffuse velocity (viscosity via Jacobi iteration)
3. Project velocity (enforce incompressibility — Member 2's step)
4. Advect velocity (self-advection, semi-Lagrangian)
5. Project again
6. Diffuse density
7. Advect density

Key constraint: NO Python loops over grid cells. All operations must be
NumPy vectorized using array slicing like grid[1:-1, 1:-1, 1:-1].

[PASTE YOUR SPECIFIC QUESTION OR CODE HERE]
```

---

---

## Member 2 — Solver Architect

**File you own:** `fluid/solver.py`

**Your job:** Improve the pressure projection solver and own the Physics/ML hybrid switch.

### What the pressure solver does

After advection, the velocity field has non-zero divergence — meaning fluid is "piling up" in some cells. The pressure solver fixes this by:

1. Computing divergence: `div = du/dx + dv/dy + dw/dz`
2. Solving the Poisson equation: `∇²p = div` (finds pressure that counteracts the divergence)
3. Subtracting the pressure gradient from velocity: `v = v - ∇p`

After this step, `divergence ≈ 0` everywhere. That's the goal.

### What's already built

`solver.py` already has:
- A working Jacobi solver (`_project_jacobi`) — 40 iterations, functional but improvable
- The hybrid switch (`mode = "PHYSICS"` or `mode = "ML"`)
- The pressure gradient subtraction (`_subtract_pressure_gradient`)
- Divergence error metrics returned from every call

### Your Week 2 task

The Jacobi solver works but converges slowly. Your job is to either:

**Option A (easier):** Tune the existing Jacobi solver. Try increasing iterations from 40 to 80 and measure if divergence drops meaningfully. Find the sweet spot between accuracy and speed.

**Option B (better):** Replace Jacobi with a **Conjugate Gradient solver using SciPy**. This is faster and more accurate. Use `scipy.sparse.linalg.cg()`.

```bash
pip install scipy
```

### The hybrid switch (Week 5)

The switch is already wired in `solver.py`. When Member 4 gives you their ONNX model:

```python
import onnxruntime as ort

session = ort.InferenceSession("pressure_model.onnx")
sim.set_mode("ML", ml_model=session)
```

That's all you need to plug it in. The `_project_ml()` function in `solver.py` handles the rest.

### How to test your solver

```bash
python main.py --mode benchmark --frames 50
```

Look at `project1_ms` and `project2_ms` in the output. Those are your solver times. Also watch `divergence_max` in headless mode — it should be as low as possible (ideally < 1.0).

### Divergence quality check

Write a quick test in a scratch file:

```python
from fluid import FluidSimulation
import numpy as np

sim = FluidSimulation(N=32)
sim.add_smoke_source(16, 2, 16, velocity=(0.5, 2.0, 0.3))

for i in range(10):
    sim.step(solver_iterations=40)
    div = sim.grid.compute_divergence()
    print(f"Frame {i}: max_div={np.abs(div).max():.6f}, mean_div={np.abs(div).mean():.8f}")
```

After a good solver run, `mean_div` should be in the range of `0.00001` to `0.001`.

### Your Claude AI prompt

```
I am Member 2 (Solver Architect) on a 3D fluid simulation project.
I own fluid/solver.py which implements pressure projection for a 32³ MAC grid fluid sim.

The solver enforces incompressibility by solving the Poisson equation:
  ∇²p = div(v)
using Jacobi iteration, then subtracting the pressure gradient from velocity.

Current Jacobi implementation:
- Iterates 40 times per frame
- Uses NumPy slicing: p[1:-1,1:-1,1:-1] = (sum of 6 neighbors - dx²*div) / 6
- Takes ~7ms per call on a ThinkBook laptop

The grid structure:
- All scalar fields (pressure, divergence): shape (32, 32, 32)
- u velocity: (33, 32, 32), v: (32, 33, 32), w: (32, 32, 33)

There is a hybrid mode switch:
  if mode == "PHYSICS": use Jacobi
  elif mode == "ML": call ONNX model

Key constraint: no Python loops over cells, NumPy only.

[PASTE YOUR SPECIFIC QUESTION OR CODE HERE]
```

---

---

## Member 3 — Data Pipeline Engineer

**File you own:** `data_pipeline.py`

**Your job:** Run the physics simulation to generate training data, and manage the `.npy` dataset that Member 4 will use to train their model.

### What the data pipeline does

The ML model needs examples of: *"here is a broken velocity field → here is the pressure that fixes it."*

Your pipeline:
1. Runs the physics simulation many times with **random smoke source positions**
2. After every frame, saves the input (velocity divergence) and output (pressure) as `.npy` files
3. Organizes them into a clean folder structure
4. Writes a `metadata.json` describing everything

### Generating the dataset

```bash
# Quick test run (5 runs, 200 frames each — ~10 minutes)
python main.py --mode data --runs 5 --frames 200

# Full dataset for training (10 runs, 500 frames — overnight)
python main.py --mode data --runs 10 --frames 500
```

Output will appear in `data/`:
```
data/
├── metadata.json
├── run_001/
│   ├── frame_0000_divergence.npy   ← ML model INPUT  (32,32,32)
│   ├── frame_0000_pressure.npy     ← ML model TARGET (32,32,32)
│   ├── frame_0000_velocity_u.npy
│   ├── frame_0000_velocity_v.npy
│   ├── frame_0000_velocity_w.npy
│   ├── frame_0000_density.npy
│   └── frame_0002_...              (saves every 2nd frame by default)
├── run_002/
│   └── ...
```

### What to hand off to Member 4

Tell Member 4:
- The path to your `data/` folder
- That `divergence.npy` is the model input (shape `32,32,32`)
- That `pressure.npy` is the model target (shape `32,32,32`)
- The total number of samples: `n_runs × (frames / save_every)`

### Extending the pipeline (Week 4)

If Member 4 needs more variety in the training data, edit `data_pipeline.py` to add:
- Different grid sizes (N=16 for fast, N=32 for standard)
- Different viscosity/diffusion values
- Multiple simultaneous smoke sources
- Obstacle placement (a solid cell block in the middle)

### Verifying the data is good

Write a quick check:

```python
import numpy as np
import os

div = np.load("data/run_001/frame_0010_divergence.npy")
prs = np.load("data/run_001/frame_0010_pressure.npy")

print(f"Divergence shape: {div.shape}")      # should be (32, 32, 32)
print(f"Pressure shape:   {prs.shape}")      # should be (32, 32, 32)
print(f"Divergence range: {div.min():.3f} to {div.max():.3f}")
print(f"Pressure range:   {prs.min():.3f} to {prs.max():.3f}")
print(f"Any NaN? div={np.isnan(div).any()}, prs={np.isnan(prs).any()}")
```

No NaNs and non-zero ranges = healthy data.

### Your Claude AI prompt

```
I am Member 3 (Data Pipeline Engineer) on a 3D fluid simulation project.
I own data_pipeline.py which generates ML training data from a physics simulator.

The simulator is a 32³ MAC grid fluid sim (Stable Fluids algorithm).
My pipeline runs FluidSimulation, calls get_dataset_snapshot() each frame,
and saves the results as .npy files.

Dataset format:
- Input (X):  divergence field, shape (32, 32, 32), dtype float32
- Target (y): pressure field,   shape (32, 32, 32), dtype float32
- Also saves: velocity_u (33,32,32), velocity_v (32,33,32), velocity_w (32,32,33), density (32,32,32)

Folder structure: data/run_NNN/frame_NNNN_<field>.npy

The FluidSimulation class interface:
  sim = FluidSimulation(N=32, dt=0.1, diffusion=0.00005, viscosity=0.00001)
  sim.add_smoke_source(x, y, z, density_rate=4.0, velocity=(vx,vy,vz), temperature=1.0)
  metrics = sim.step(solver_iterations=40)
  snapshot = sim.get_dataset_snapshot()  # returns dict of numpy arrays

[PASTE YOUR SPECIFIC QUESTION OR CODE HERE]
```

---

---

## Member 4 — Model Architect

**Files you create:** `model.py`, `train.py`, `export_onnx.py`

**Your job:** Design a lightweight 3D CNN that learns to predict the pressure field from the divergence field, train it on Member 3's data, and export it to ONNX for fast CPU inference.

### The ML problem

```
Input:  divergence field  — shape (32, 32, 32)  — "the broken velocity"
Output: pressure field    — shape (32, 32, 32)  — "what fixes it"
```

This is a **3D regression problem** — same shape in, same shape out.

### Recommended architecture: Mini 3D U-Net

A shallow U-Net works well here. It has encoder (downsampling) and decoder (upsampling) paths with skip connections.

```
Input (1, 32, 32, 32)
  → Conv3D(1→8, kernel=3)   + ReLU
  → Conv3D(8→16, kernel=3)  + ReLU   ← encoder
  → MaxPool3D(2)             → (16, 16, 16, 16)
  → Conv3D(16→32, kernel=3) + ReLU
  → Upsample(2)              → (32, 32, 32, 32)  [skip connection here]
  → Conv3D(32→16, kernel=3) + ReLU   ← decoder
  → Conv3D(16→1, kernel=1)           ← output layer
Output (1, 32, 32, 32)
```

Keep it **3–4 layers total**. Deeper = slower inference. You must hit **<200ms on CPU**.

### Training setup

```bash
pip install torch torchvision
```

Loss function: **Mean Squared Error (MSE)** between predicted pressure and true pressure.

```python
loss = F.mse_loss(predicted_pressure, true_pressure)
```

Also track **L2 relative error** — this is the metric Member 6 will put in the final report:
```python
l2_error = torch.norm(pred - target) / torch.norm(target)
```

### Exporting to ONNX

Once trained, export for fast CPU inference (no PyTorch overhead at runtime):

```python
import torch
torch.onnx.export(
    model,
    dummy_input,                    # shape: (1, 1, 32, 32, 32)
    "pressure_model.onnx",
    input_names=["divergence_input"],
    output_names=["pressure_output"],
    opset_version=11
)
```

### Testing inference speed

```python
import onnxruntime as ort
import numpy as np
import time

session = ort.InferenceSession("pressure_model.onnx")
x = np.random.randn(1, 1, 32, 32, 32).astype(np.float32)

# Warm up
for _ in range(5):
    session.run(None, {"divergence_input": x})

# Benchmark
times = []
for _ in range(50):
    t0 = time.perf_counter()
    session.run(None, {"divergence_input": x})
    times.append((time.perf_counter() - t0) * 1000)

print(f"Mean inference: {np.mean(times):.1f}ms")  # must be < 200ms
```

### Handing off to Member 2

Give Member 2 your `pressure_model.onnx` file. They plug it in with:

```python
import onnxruntime as ort
session = ort.InferenceSession("pressure_model.onnx")
sim.set_mode("ML", ml_model=session)
```

### Your Claude AI prompt

```
I am Member 4 (Model Architect) on a 3D fluid simulation project.
I need to design and train a lightweight 3D CNN to approximate a fluid pressure solver.

Problem:
  Input:  divergence field, shape (32, 32, 32), dtype float32
            — represents "how broken" the velocity field is
  Output: pressure field,   shape (32, 32, 32), dtype float32
            — the correction that makes velocity divergence-free

Training data: .npy files generated by a physics simulator
  X = np.load("data/run_001/frame_0000_divergence.npy")  # (32,32,32)
  y = np.load("data/run_001/frame_0000_pressure.npy")    # (32,32,32)

Hard constraints:
  - Model must run in < 200ms on CPU (no GPU at demo time)
  - Use PyTorch for training, export to ONNX for inference
  - ONNX input name must be "divergence_input", shape (1, 1, 32, 32, 32)
  - Keep architecture shallow: 3–4 layers max

Metrics to report:
  - L2 relative error vs physics baseline
  - Inference time in ms (ONNX runtime, CPU)
  - FPS improvement over classical Jacobi solver (~53ms/frame baseline)

[PASTE YOUR SPECIFIC QUESTION OR CODE HERE]
```

---

---

## Member 5 — Visualization Specialist

**File you own:** `visualizer.py`

**Your job:** Build the real-time viewer that shows the simulation running, with a toggle button to switch between Physics and ML mode live.

### What's already built

`visualizer.py` has a working `FluidVisualizer` class with:
- A 3-subplot figure (XY, XZ, YZ slices of the density field)
- `FuncAnimation` loop that steps the sim and updates the plots
- A custom smoke colormap (black → orange → white)

The issue: it needs `python3-tk` to show an interactive window.

```bash
sudo apt install python3-tk
python main.py --mode live --N 32
```

### Your Week 2–3 task

Extend the visualizer with:

**1. Mode toggle button**

```python
from matplotlib.widgets import Button

ax_btn = fig.add_axes([0.45, 0.01, 0.1, 0.04])
btn = Button(ax_btn, 'PHYSICS', color='#1a1a2e', hovercolor='#7c6af7')

def toggle_mode(event):
    if sim.mode == "PHYSICS":
        sim.set_mode("ML", ml_model=onnx_session)
        btn.label.set_text("ML")
    else:
        sim.set_mode("PHYSICS")
        btn.label.set_text("PHYSICS")

btn.on_clicked(toggle_mode)
```

**2. FPS counter in the title**

Already in the base code — just make it more visible.

**3. Velocity arrows (optional but impressive)**

```python
# Show velocity direction as arrows on the center slice
uc, vc, wc = sim.grid.get_velocity_at_center()
ax.quiver(uc[:, :, 16][::4, ::4], vc[:, :, 16][::4, ::4], scale=20, color='cyan', alpha=0.4)
```

### How to read the grid for rendering

```python
grid = sim.grid
N = grid.N
mid = N // 2

# 3 slices of the density field
xy_slice = grid.density[:, :, mid]   # shape (32, 32) — top-down
xz_slice = grid.density[:, mid, :]   # shape (32, 32) — front view
yz_slice = grid.density[mid, :, :]   # shape (32, 32) — side view
```

These are just 2D NumPy arrays — pass them to `imshow()`.

### Running the visualizer standalone

```bash
sudo apt install python3-tk   # only needed once
source .venv/bin/activate
python main.py --mode live --N 32
```

### Your Claude AI prompt

```
I am Member 5 (Visualization Specialist) on a 3D fluid simulation project.
I own visualizer.py which displays a real-time 3D fluid simulation.

The simulation is a 32³ MAC grid fluid sim. I access it via:
  from fluid import FluidSimulation
  sim = FluidSimulation(N=32)
  sim.add_smoke_source(x, y, z, density_rate=3.0, velocity=(0,1.5,0))
  metrics = sim.step()

  # Reading the density field for rendering:
  grid = sim.grid
  xy_slice = grid.density[:, :, 16]   # shape (32, 32)
  xz_slice = grid.density[:, 16, :]   # shape (32, 32)
  yz_slice = grid.density[16, :, :]   # shape (32, 32)

  # Getting cell-centered velocity:
  uc, vc, wc = grid.get_velocity_at_center()  # each shape (32, 32, 32)

  # Switching modes:
  sim.set_mode("PHYSICS")
  sim.set_mode("ML", ml_model=onnx_session)

Current tech: matplotlib with FuncAnimation.
I need to add: live mode toggle button, FPS display, and optionally velocity arrows.

Platform: Ubuntu Linux, ThinkBook laptop, python3-tk installed.

[PASTE YOUR SPECIFIC QUESTION OR CODE HERE]
```

---

---

## Member 6 — System Integrator

**File you own:** `main.py`

**Your job:** Wire everything together into a single working application, and run the final benchmarks that go in the report.

### What's already in main.py

Four run modes already work:
- `--mode headless` — physics loop, no display
- `--mode live` — calls visualizer
- `--mode benchmark` — per-step timing breakdown
- `--mode data` — dataset generation

### Your Week 6 task: integrate the ONNX model

When Member 4 hands you `pressure_model.onnx`, update `main.py`:

```python
def run_live(N=32, model_path=None):
    from fluid import FluidSimulation
    from visualizer import FluidVisualizer
    import onnxruntime as ort

    sim = FluidSimulation(N=N)

    if model_path:
        session = ort.InferenceSession(model_path)
        sim.set_mode("ML", ml_model=session)
        print("ML mode active")

    viz = FluidVisualizer(sim)
    viz.run()
```

Add a `--model` argument:

```python
parser.add_argument("--model", type=str, default=None,
                    help="Path to ONNX model (enables ML mode)")
```

Then run both modes:

```bash
# Physics mode
python main.py --mode live --N 32

# ML mode
python main.py --mode live --N 32 --model pressure_model.onnx
```

### Week 7 task: final benchmarks

Generate the comparison data for the report:

```python
def run_final_benchmark(model_path: str, N=32, frames=100):
    from fluid import FluidSimulation, MODE_PHYSICS, MODE_ML
    import onnxruntime as ort
    import numpy as np

    # Physics baseline
    sim_p = FluidSimulation(N=N)
    physics_times = []
    for _ in range(frames):
        sim_p.add_smoke_source(N//2, 2, N//2)
        m = sim_p.step()
        physics_times.append(m["total_ms"])

    # ML mode
    session = ort.InferenceSession(model_path)
    sim_ml = FluidSimulation(N=N)
    sim_ml.set_mode(MODE_ML, ml_model=session)
    ml_times = []
    for _ in range(frames):
        sim_ml.add_smoke_source(N//2, 2, N//2)
        m = sim_ml.step()
        ml_times.append(m["total_ms"])

    speedup = np.mean(physics_times) / np.mean(ml_times)
    print(f"Physics: {np.mean(physics_times):.1f}ms ({1000/np.mean(physics_times):.1f} FPS)")
    print(f"ML:      {np.mean(ml_times):.1f}ms ({1000/np.mean(ml_times):.1f} FPS)")
    print(f"Speedup: {speedup:.1f}x")
```

### Generating accuracy charts (Week 7)

For the L2 error chart comparing ML pressure vs physics pressure:

```python
import matplotlib.pyplot as plt

# Collect per-frame L2 errors from Member 4's evaluation script
# Plot them:
plt.figure(figsize=(10, 4))
plt.plot(l2_errors)
plt.xlabel("Frame")
plt.ylabel("L2 Relative Error")
plt.title("ML Pressure Prediction Error vs Physics Baseline")
plt.axhline(y=0.05, color='r', linestyle='--', label='5% threshold')
plt.legend()
plt.savefig("accuracy_chart.png", dpi=150)
```

### Your Claude AI prompt

```
I am Member 6 (System Integrator) on a 3D fluid simulation project.
I own main.py and am responsible for wiring physics + ML + visualization together.

The core simulation interface:
  from fluid import FluidSimulation, MODE_PHYSICS, MODE_ML
  sim = FluidSimulation(N=32, dt=0.1, diffusion=0.00005, viscosity=0.00001)
  sim.add_smoke_source(x, y, z, density_rate=3.0, velocity=(0,1.5,0), temperature=0.8)
  metrics = sim.step(solver_iterations=40)
  # metrics contains: total_ms, fps, divergence_max, density_total, per-step timings

  # Mode switching:
  sim.set_mode("PHYSICS")
  sim.set_mode("ML", ml_model=onnx_session)  # onnx_session from onnxruntime

  # Performance log:
  sim.perf_log   # list of metrics dicts, one per frame

The visualizer:
  from visualizer import FluidVisualizer
  viz = FluidVisualizer(sim)
  viz.run(fps=10)

I need to:
1. Add --model CLI argument to load an ONNX model
2. Generate side-by-side FPS benchmark comparing PHYSICS vs ML modes
3. Generate L2 error accuracy charts
4. Make a clean final benchmark report

[PASTE YOUR SPECIFIC QUESTION OR CODE HERE]
```

---

---

## How Everything Connects

```
                    ┌─────────────────────────────────────────┐
                    │           EVERY FRAME                   │
                    │                                         │
  Member 1          │   grid.py → advect.py → diffuse.py     │
  (Physics Core)    │   forces.py → simulation.py            │
                    │              ↓                          │
  Member 2          │         solver.py                       │
  (Solver)          │    [PHYSICS] Jacobi iteration           │
                    │    [ML]      ONNX inference ←────────── Member 4
                    │              ↓                          │  (Model)
  Member 3          │   get_dataset_snapshot()                │
  (Data)            │   → saves .npy files ──────────────────→│
                    │              ↓                          │
  Member 5          │       visualizer.py                     │
  (Viz)             │   renders density slices + mode toggle  │
                    │              ↓                          │
  Member 6          │          main.py                        │
  (Integrator)      │   orchestrates everything + benchmarks  │
                    └─────────────────────────────────────────┘
```

**Dependency order:**
1. Member 1's code must work before anyone else can start
2. Member 2 and Member 3 can work in parallel once Member 1 is done
3. Member 4 needs Member 3's data
4. Member 5 can start the UI anytime (just needs Member 1's grid)
5. Member 6 integrates everything in Week 5–6

---

## Timeline

| Week | Member 1 | Member 2 | Member 3 | Member 4 | Member 5 | Member 6 |
|------|----------|----------|----------|----------|----------|----------|
| 1 | ✅ Grid done | Study Jacobi | Set up pipeline | Design CNN arch | Build 3-slice viewer | Study main.py |
| 2 | ✅ Advection | Improve solver | Test data gen | Write training script | Integrate with sim | — |
| 3 | ✅ Diffusion + debug | Benchmark solver | Generate full dataset | — | Add velocity arrows | — |
| 4 | Support others | — | Run overnight data gen | Train model on data | — | — |
| 5 | — | Plug in ONNX | — | Export to ONNX | Add mode toggle | Wire everything |
| 6 | — | — | — | Tune for <200ms | Polish UI | Run final benchmarks |
| 7 | — | — | — | Calculate L2 error | — | Generate charts |
| 8 | Report | Report | Report | Report | Record demo | Report |

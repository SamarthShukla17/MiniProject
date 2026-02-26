# 3D Fluid Simulation — Squad A Physics Engine

A 32³ MAC grid fluid simulator implementing the "Stable Fluids" algorithm (Jos Stam, 1999).
Built for CPU with NumPy vectorization. No GPU required.

---

## Project Structure

```
fluid_sim/
├── main.py               ← Entry point (run this)
├── visualizer.py         ← 3-slice matplotlib viewer (Squad C)
├── data_pipeline.py      ← ML training data generator (Member 3)
├── requirements.txt
└── fluid/
    ├── __init__.py
    ├── grid.py           ← MAC staggered grid (Week 1)
    ├── advect.py         ← Semi-Lagrangian advection (Week 2)
    ├── diffuse.py        ← Jacobi diffusion / viscosity (Week 3)
    ├── forces.py         ← Buoyancy, wind, heat sources (Week 3)
    ├── simulation.py     ← Master physics loop
    └── solver.py         ← Pressure projection + ML hybrid switch
```

---

## Setup

### Requirements
- Python 3.10 or higher
- pip

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

That's it. Only NumPy and Matplotlib are needed.

---

## Running the Simulation

All modes are accessed through `main.py`:

### 1. Headless mode (no display — best for servers/SSH)
Runs the simulation and prints performance stats to the terminal.

```bash
python main.py --mode headless --N 32 --frames 100
```

Expected output:
```
Headless simulation | N=32 | 100 frames
────────────────────────────────────────
  Frame 000 |  74.9ms (13.3 FPS) | div_max=99.6 | density=375.6
  Frame 010 |  38.5ms (26.0 FPS) | div_max=98.6 | density=3940.8
  ...
  Average: 52ms/frame (19 FPS)
```

---

### 2. Live visualization (requires a display)
Opens a real-time window with 3 orthogonal slices (XY, XZ, YZ) of the smoke density.

```bash
python main.py --mode live --N 32
```

> **Note:** Requires a desktop environment (won't work over plain SSH).
> On a remote server use `--mode headless` instead.

---

### 3. Benchmark mode
Detailed breakdown of time spent in each physics step.
This is the **"Physics Baseline"** number the ML model must beat.

```bash
python main.py --mode benchmark --N 32 --frames 50
```

Expected output:
```
Step               Mean     Min     Max
────────────────────────────────────────────────
  forces_ms        0.2ms   0.1ms   0.2ms
  diffuse_vel_ms  12.8ms  11.2ms  22.3ms
  project1_ms      7.4ms   6.1ms  18.3ms
  advect_vel_ms   17.1ms  10.4ms  30.6ms
  project2_ms      7.5ms   6.2ms  18.6ms
  diffuse_den_ms   4.0ms   3.4ms  13.8ms
  advect_den_ms    3.8ms   2.8ms  14.6ms
  total_ms        52.9ms  41.1ms  64.5ms

FPS (physics only): 18.9
⚡ ML target: < 53ms/frame to match physics
🏆 ML goal:   < 5ms/frame for 10x speedup
```

---

### 4. Data generation mode (for Squad B / ML team)
Runs multiple simulation runs with randomized parameters and saves
velocity + pressure snapshots as `.npy` files for ML training.

```bash
python main.py --mode data --N 32 --runs 5 --frames 200
```

Output structure:
```
data/
├── metadata.json
├── run_001/
│   ├── frame_0000_divergence.npy   ← ML INPUT  (32,32,32)
│   ├── frame_0000_pressure.npy     ← ML TARGET (32,32,32)
│   ├── frame_0000_velocity_u.npy   ← (33,32,32)
│   ├── frame_0000_velocity_v.npy   ← (32,33,32)
│   ├── frame_0000_velocity_w.npy   ← (32,32,33)
│   ├── frame_0000_density.npy      ← (32,32,32)
│   └── ...
├── run_002/
│   └── ...
```

Loading in Python (for Member 4):
```python
import numpy as np
X = np.load("data/run_001/frame_0000_divergence.npy")  # input
y = np.load("data/run_001/frame_0000_pressure.npy")    # target
```

---

## Command Line Arguments

| Argument   | Default    | Description                          |
|------------|------------|--------------------------------------|
| `--mode`   | `headless` | `live`, `headless`, `benchmark`, `data` |
| `--N`      | `32`       | Grid resolution (32 = 32³ cells)     |
| `--frames` | `100`      | Number of frames to simulate         |
| `--runs`   | `5`        | Number of data generation runs       |

---

## For Other Squad Members

### Member 2 (Solver Architect)
Your work lives in `fluid/solver.py`. The Jacobi solver is already implemented.
You can tune `iterations` in `_project_jacobi()` or swap in a Conjugate Gradient solver.
The hybrid switch is already wired:
```python
# In solver.py — plug your ML model into this branch:
elif mode == "ML":
    pressure = _project_ml(grid, ml_model)
```

### Member 3 (Data Pipeline)
Use `data_pipeline.py` directly or via `python main.py --mode data`.
The `sim.get_dataset_snapshot()` method returns a dict of all arrays per frame.

### Member 4 (ML / ONNX)
Load `.npy` files from `data/`. Input is `divergence.npy` (32,32,32), target is `pressure.npy` (32,32,32).
When your ONNX model is ready, pass it to:
```python
sim.set_mode("ML", ml_model=onnx_session)
```

### Member 5 (Visualization)
Extend `visualizer.py`. The `FluidVisualizer` class is yours to build on.
Add your mode toggle button to the `_setup_figure()` method.

### Member 6 (System Integrator)
Extend `main.py`. Use `FluidSimulation.set_mode("ML", model)` to switch at runtime.
All per-frame metrics are in `sim.perf_log` for your benchmarking charts.

---

## Physics Notes

- **Grid:** MAC (Marker-and-Cell) staggered grid. Pressure at cell centers, velocity on faces.
- **Advection:** Semi-Lagrangian with trilinear interpolation (unconditionally stable).
- **Diffusion:** Implicit Jacobi iteration (unconditionally stable).
- **Projection:** Jacobi pressure solver enforces div(v) = 0 (incompressibility).
- **Buoyancy:** Hot smoke rises automatically via temperature-driven upward force.

Reference: Jos Stam, *"Stable Fluids"*, SIGGRAPH 1999.

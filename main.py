"""
main.py — Master Entry Point (Member 6's Domain)
=================================================
This is the top-level script that runs everything.
Member 6 (System Integrator) will expand this with the full
Physics/ML toggle UI, but this version runs a complete demo.

Usage:
    python main.py                    # Live visualization (default)
    python main.py --mode data        # Generate ML training dataset
    python main.py --mode benchmark   # Benchmark physics performance
    python main.py --mode headless    # Run without display (for servers)
"""

import argparse
import numpy as np
import time


def run_live(N: int = 32):
    """Live interactive visualization."""
    from fluid import FluidSimulation
    from visualizer import FluidVisualizer

    print(f"Starting live simulation (N={N})...")
    print("Close the window to exit.\n")

    sim = FluidSimulation(N=N, dt=0.1, diffusion=0.00005, viscosity=0.00001)
    viz = FluidVisualizer(sim)
    viz.run(fps=10)


def run_headless(N: int = 32, frames: int = 100):
    """Run simulation without display — prints stats each frame."""
    from fluid import FluidSimulation

    print(f"\nHeadless simulation | N={N} | {frames} frames")
    print(f"{'─'*60}")

    sim = FluidSimulation(N=N, dt=0.1, diffusion=0.00005, viscosity=0.00001)
    total_times = []

    for f in range(frames):
        # Smoke source at bottom center
        sim.add_smoke_source(N // 2, 2, N // 2,
                             density_rate=3.0,
                             velocity=(0.0, 1.5, 0.0),
                             temperature=0.8)

        metrics = sim.step(solver_iterations=20)
        total_times.append(metrics["total_ms"])

        if f % 10 == 0:
            print(f"  Frame {f:03d} | {metrics['total_ms']:6.1f}ms "
                  f"({metrics['fps']:.1f} FPS) | "
                  f"div_max={metrics['divergence_max']:.5f} | "
                  f"density={metrics['density_total']:.1f}")

    print(f"\n{'─'*60}")
    print(f"  Average: {np.mean(total_times):.1f}ms/frame ({1000/np.mean(total_times):.1f} FPS)")
    print(f"  Min:     {np.min(total_times):.1f}ms")
    print(f"  Max:     {np.max(total_times):.1f}ms")
    print(f"  Solver target to beat for ML: {np.mean(total_times):.0f}ms")


def run_benchmark(N: int = 32, frames: int = 50):
    """
    Detailed performance breakdown.
    Shows how long each physics step takes.
    This is the baseline the ML model must beat.
    """
    from fluid import FluidSimulation

    print(f"\n{'='*60}")
    print(f"  PHYSICS BASELINE BENCHMARK | N={N} | {frames} frames")
    print(f"{'='*60}")

    sim = FluidSimulation(N=N, dt=0.1, diffusion=0.00005, viscosity=0.00001)

    # Warm up
    for _ in range(5):
        sim.add_smoke_source(N//2, 2, N//2)
        sim.step(solver_iterations=20)

    # Benchmark
    logs = []
    for f in range(frames):
        sim.add_smoke_source(N//2, 2, N//2,
                             density_rate=3.0,
                             velocity=(0.0, 1.5, 0.0),
                             temperature=0.8)
        metrics = sim.step(solver_iterations=40)
        logs.append(metrics)

    # Summary
    keys = ["forces_ms", "diffuse_vel_ms", "project1_ms",
            "advect_vel_ms", "project2_ms", "diffuse_den_ms",
            "advect_den_ms", "total_ms"]

    print(f"\n{'Step':<20} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print(f"{'─'*50}")
    for k in keys:
        vals = [m[k] for m in logs]
        print(f"  {k:<18} {np.mean(vals):>7.1f}ms {np.min(vals):>7.1f}ms {np.max(vals):>7.1f}ms")

    total_vals = [m["total_ms"] for m in logs]
    print(f"\n{'─'*50}")
    print(f"  FPS (physics only): {1000/np.mean(total_vals):.1f}")
    print(f"\n  ⚡ ML model target: < {np.mean(total_vals):.0f}ms/frame to match physics")
    print(f"  🏆 ML model goal:   < {np.mean(total_vals)*0.1:.0f}ms/frame for 10x speedup")


def run_data_generation(N: int = 32, n_runs: int = 5, frames: int = 200):
    """Generate training dataset for Squad B (ML team)."""
    from data_pipeline import DatasetGenerator

    print(f"\nData generation mode")
    print(f"  N={N}, {n_runs} runs, {frames} frames/run")
    print(f"  Saving to: ./data/\n")

    gen = DatasetGenerator(output_dir="data", N=N)
    gen.generate_dataset(
        n_runs=n_runs,
        frames_per_run=frames,
        save_every=2
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Fluid Simulation")
    parser.add_argument(
        "--mode", choices=["live", "headless", "benchmark", "data"],
        default="headless",
        help="Run mode (default: headless)"
    )
    parser.add_argument("--N",      type=int, default=32, help="Grid resolution (default: 32)")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames")
    parser.add_argument("--runs",   type=int, default=5,   help="Number of data gen runs")

    args = parser.parse_args()

    if args.mode == "live":
        run_live(N=args.N)
    elif args.mode == "headless":
        run_headless(N=args.N, frames=args.frames)
    elif args.mode == "benchmark":
        run_benchmark(N=args.N, frames=args.frames)
    elif args.mode == "data":
        run_data_generation(N=args.N, n_runs=args.runs, frames=args.frames)

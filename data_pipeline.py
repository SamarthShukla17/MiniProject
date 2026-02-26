"""
data_pipeline.py — Training Dataset Generator (Member 3's Domain)
==================================================================
Captures simulation states and saves them as .npy files for ML training.

Dataset structure on disk:
  data/
    run_001/
      frame_0000_velocity_u.npy    ← shape (33, 32, 32)
      frame_0000_velocity_v.npy    ← shape (32, 33, 32)
      frame_0000_velocity_w.npy    ← shape (32, 32, 33)
      frame_0000_divergence.npy    ← shape (32, 32, 32)  ← ML INPUT
      frame_0000_pressure.npy      ← shape (32, 32, 32)  ← ML TARGET
      frame_0000_density.npy       ← shape (32, 32, 32)
      ...
    run_002/
      ...
    metadata.json                  ← grid params, total frames, etc.

ML team (Member 4) loads this with:
  X = np.load("data/run_001/frame_0000_divergence.npy")
  y = np.load("data/run_001/frame_0000_pressure.npy")
"""

import numpy as np
import json
import os
from pathlib import Path
from fluid import FluidSimulation


class DatasetGenerator:
    """
    Runs the physics simulation and captures snapshots for ML training.

    Usage:
        gen = DatasetGenerator(output_dir="data/", N=32)
        gen.generate_run(
            run_id=1,
            n_frames=500,
            random_seed=42
        )
    """

    def __init__(self, output_dir: str = "data", N: int = 32):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.N = N
        self.metadata = {
            "N": N,
            "runs": []
        }

    def generate_run(
        self,
        run_id: int,
        n_frames: int = 500,
        random_seed: int = None,
        dt: float = 0.1,
        diffusion: float = 0.00005,
        viscosity: float = 0.00001,
        n_sources: int = 1,
        solver_iterations: int = 40,
        save_every: int = 1,
    ):
        """
        Generate one full simulation run and save all frames.

        Args:
            run_id           : Integer ID for this run (used in folder name)
            n_frames         : How many physics steps to simulate
            random_seed      : For reproducibility (randomizes source positions)
            dt               : Timestep
            diffusion        : Smoke diffusion rate
            viscosity        : Fluid viscosity
            n_sources        : Number of smoke emitters (1–4 for variety)
            solver_iterations: Jacobi iterations per frame
            save_every       : Save a snapshot every N frames (1 = all frames)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        run_dir = self.output_dir / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize fresh simulation
        sim = FluidSimulation(N=self.N, dt=dt,
                              diffusion=diffusion, viscosity=viscosity)

        # Randomize smoke source positions for dataset variety
        N = self.N
        sources = []
        for _ in range(n_sources):
            x = np.random.randint(N // 4, 3 * N // 4)
            y = np.random.randint(1, N // 4)          # near the bottom
            z = np.random.randint(N // 4, 3 * N // 4)
            vel_x = np.random.uniform(-0.5, 0.5)
            vel_z = np.random.uniform(-0.5, 0.5)
            sources.append((x, y, z, vel_x, vel_z))

        print(f"\n[DataGen] Starting run {run_id:03d} | "
              f"{n_frames} frames | {n_sources} source(s) | seed={random_seed}")

        saved_count = 0
        for frame in range(n_frames):
            # Inject smoke from each source every frame
            for (sx, sy, sz, vx, vz) in sources:
                sim.add_smoke_source(
                    sx, sy, sz,
                    density_rate=4.0,
                    velocity=(vx, 2.0, vz),
                    temperature=1.0
                )

            # Step physics
            metrics = sim.step(solver_iterations=solver_iterations)

            # Save snapshot every `save_every` frames
            if frame % save_every == 0:
                snapshot = sim.get_dataset_snapshot()
                prefix = run_dir / f"frame_{frame:04d}"

                np.save(f"{prefix}_velocity_u.npy",  snapshot["velocity_u"])
                np.save(f"{prefix}_velocity_v.npy",  snapshot["velocity_v"])
                np.save(f"{prefix}_velocity_w.npy",  snapshot["velocity_w"])
                np.save(f"{prefix}_divergence.npy",  snapshot["divergence"])
                np.save(f"{prefix}_pressure.npy",    snapshot["pressure"])
                np.save(f"{prefix}_density.npy",     snapshot["density"])

                saved_count += 1

            # Progress update every 50 frames
            if frame % 50 == 0:
                print(f"  Frame {frame:04d}/{n_frames} | "
                      f"{metrics['fps']:.1f} FPS | "
                      f"div_max={metrics['divergence_max']:.5f} | "
                      f"density={metrics['density_total']:.1f}")

        print(f"[DataGen] Run {run_id:03d} done. Saved {saved_count} snapshots → {run_dir}")

        # Save run metadata
        run_meta = {
            "run_id"       : run_id,
            "n_frames"     : n_frames,
            "saved_frames" : saved_count,
            "save_every"   : save_every,
            "random_seed"  : random_seed,
            "N"            : N,
            "dt"           : dt,
            "diffusion"    : diffusion,
            "viscosity"    : viscosity,
            "n_sources"    : n_sources,
            "sources"      : sources,
            "directory"    : str(run_dir),
        }
        self.metadata["runs"].append(run_meta)

        # Write metadata JSON
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return run_meta

    def generate_dataset(
        self,
        n_runs: int = 10,
        frames_per_run: int = 500,
        save_every: int = 2,
    ):
        """
        Generate a full multi-run dataset for ML training.
        
        Each run uses a different random seed for source positions,
        giving the model a diverse set of fluid behaviors to learn from.
        
        Total dataset size estimate:
          - 10 runs × 250 frames × 6 arrays × (32³ float32)
          - ≈ 10 × 250 × 6 × 131,072 bytes ≈ ~1.9 GB
          - With save_every=2: ~0.95 GB

        Args:
            n_runs         : Number of simulation runs
            frames_per_run : Frames per run
            save_every     : Save every Nth frame (reduce disk usage)
        """
        print(f"\n{'='*60}")
        print(f"  Generating dataset: {n_runs} runs × {frames_per_run} frames")
        print(f"  Saving every {save_every} frames → ~{n_runs * frames_per_run // save_every} total snapshots")
        print(f"{'='*60}")

        for run_id in range(1, n_runs + 1):
            self.generate_run(
                run_id=run_id,
                n_frames=frames_per_run,
                random_seed=run_id * 42,
                n_sources=np.random.randint(1, 4),
                save_every=save_every,
            )

        print(f"\n✓ Dataset complete. Metadata: {self.output_dir / 'metadata.json'}")
        print(f"  Total runs: {len(self.metadata['runs'])}")

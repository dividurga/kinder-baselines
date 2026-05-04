"""Isolate URDF load/unload time for box vs vhacd collision geometry.

Spawns obstacles at fixed, pre-validated non-overlapping positions so there
is zero rejection-sampling variance. Measures only p.loadURDF + p.removeBody
across many repetitions.

Usage:
    python bench_load.py
    python bench_load.py --n_obstacles 6 --n_reps 200 --object 025_mug
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as p

ASSETS_DIR = Path(__file__).parent / "assets"

# Fixed grid of positions well inside the robot workspace.
# Spaced 0.20m apart so no YCB object (max ~0.15m wide) overlaps its neighbour.
GRID_POSITIONS = [
    (x, y, 0.55)
    for x in np.arange(0.20, 0.65, 0.20)
    for y in np.arange(0.15, 0.80, 0.20)
]  # 3×3 = 9 slots — enough for any n_obstacles ≤ 9


def load_obstacles(
    client: int,
    urdf_path: str,
    positions: list[tuple],
) -> list[int]:
    ids = []
    for pos in positions:
        body_id = p.loadURDF(
            urdf_path,
            basePosition=pos,
            baseOrientation=(0, 0, 0, 1),
            useFixedBase=True,
            physicsClientId=client,
        )
        ids.append(body_id)
    return ids


def remove_obstacles(client: int, ids: list[int]) -> None:
    for body_id in ids:
        p.removeBody(body_id, physicsClientId=client)


def bench(
    mode: str,
    object_name: str,
    n_obstacles: int,
    n_reps: int,
) -> dict:
    urdf_path = str(ASSETS_DIR / "urdf" / f"{object_name}_{mode}.urdf")
    positions = GRID_POSITIONS[:n_obstacles]

    client = p.connect(p.DIRECT)

    load_times = []
    remove_times = []

    for _ in range(n_reps):
        t0 = time.perf_counter()
        ids = load_obstacles(client, urdf_path, positions)
        load_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        remove_obstacles(client, ids)
        remove_times.append(time.perf_counter() - t0)

    p.disconnect(client)

    return {
        "load_mean_ms": np.mean(load_times) * 1000,
        "load_median_ms": np.median(load_times) * 1000,
        "load_min_ms": np.min(load_times) * 1000,
        "load_max_ms": np.max(load_times) * 1000,
        "remove_mean_ms": np.mean(remove_times) * 1000,
        "reset_cycle_mean_ms": (np.mean(load_times) + np.mean(remove_times)) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_obstacles", type=int, default=6)
    parser.add_argument("--n_reps", type=int, default=200)
    parser.add_argument("--object", default="025_mug")
    parser.add_argument("--geometry_modes", nargs="+", default=["box", "vhacd"])
    args = parser.parse_args()

    assert args.n_obstacles <= len(GRID_POSITIONS), (
        f"Max {len(GRID_POSITIONS)} obstacles for pre-defined grid"
    )

    print(f"object={args.object}  n_obstacles={args.n_obstacles}  n_reps={args.n_reps}")
    print(f"{'='*55}")

    results = {}
    for mode in args.geometry_modes:
        print(f"Benchmarking {mode}...", end=" ", flush=True)
        results[mode] = bench(mode, args.object, args.n_obstacles, args.n_reps)
        r = results[mode]
        print(
            f"load={r['load_mean_ms']:.2f}ms  "
            f"remove={r['remove_mean_ms']:.2f}ms  "
            f"cycle={r['reset_cycle_mean_ms']:.2f}ms"
        )

    print(f"\n{'='*55}")
    print("SUMMARY (mean over reps, loading only)")
    print(f"{'='*55}")
    for mode, r in results.items():
        print(
            f"  {mode:6s}  load={r['load_mean_ms']:6.2f}ms "
            f"(min={r['load_min_ms']:.2f}  max={r['load_max_ms']:.2f})  "
            f"remove={r['remove_mean_ms']:.2f}ms"
        )

    if "box" in results and "vhacd" in results:
        box_load = results["box"]["load_mean_ms"]
        vhacd_load = results["vhacd"]["load_mean_ms"]
        box_cycle = results["box"]["reset_cycle_mean_ms"]
        vhacd_cycle = results["vhacd"]["reset_cycle_mean_ms"]
        print(f"\nLoad overhead  (vhacd / box): {vhacd_load / box_load:.2f}×  ({vhacd_load - box_load:+.2f}ms per reset)")
        print(f"Cycle overhead (vhacd / box): {vhacd_cycle / box_cycle:.2f}×  ({vhacd_cycle - box_cycle:+.2f}ms per reset)")
        per_million_extra_hours = (vhacd_cycle - box_cycle) * 1e6 / 1000 / 3600
        print(f"Extra wall-clock time over 1M resets: {(vhacd_cycle - box_cycle) * 1e6 / 1000:.0f}s  (~{per_million_extra_hours:.1f} hours)")


if __name__ == "__main__":
    main()

"""Benchmark env.reset() time for box vs vhacd collision geometry.

Measures how long it takes to load and place obstacles at each reset —
a cost that compounds across millions of training rollouts.

Usage:
    python bench_reset.py
    python bench_reset.py --n_resets 100 --n_obstacles 6
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kinder.envs.kinematic3d.geom_fidelity_reach3d import (
    GeomFidelityReach3DConfig,
    ObjectCentricGeomFidelityReach3DEnv,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = Path(__file__).parent / "assets"


def bench_mode(
    mode: str,
    n_resets: int,
    n_obstacles: int,
    obstacle_names: tuple[str, ...],
    base_seed: int,
) -> list[float]:
    config = GeomFidelityReach3DConfig(
        geometry_mode=mode,
        num_obstacles=n_obstacles,
        obstacle_names=obstacle_names,
        assets_dir=ASSETS_DIR,
    )
    env = ObjectCentricGeomFidelityReach3DEnv(config=config)

    times = []
    for i in range(n_resets):
        t0 = time.perf_counter()
        env.reset(seed=base_seed + i)
        times.append(time.perf_counter() - t0)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{n_resets}] last={times[-1]*1000:.1f}ms  mean={np.mean(times)*1000:.1f}ms")

    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_resets", type=int, default=50)
    parser.add_argument("--n_obstacles", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--obstacle_names", nargs="+", default=["025_mug"],
    )
    parser.add_argument("--geometry_modes", nargs="+", default=["box", "vhacd"])
    args = parser.parse_args()

    obstacle_names = tuple(args.obstacle_names)
    all_rows = []

    for mode in args.geometry_modes:
        print(f"\n{'='*50}")
        print(f"mode={mode}  obstacles={args.n_obstacles}  resets={args.n_resets}")
        print(f"{'='*50}")
        times = bench_mode(mode, args.n_resets, args.n_obstacles, obstacle_names, args.seed)
        for i, t in enumerate(times):
            all_rows.append({"geometry_mode": mode, "reset_idx": i, "reset_time_s": t})
        mean_ms = np.mean(times) * 1000
        median_ms = np.median(times) * 1000
        print(f"\n  mean={mean_ms:.1f}ms  median={median_ms:.1f}ms  min={min(times)*1000:.1f}ms  max={max(times)*1000:.1f}ms")

    df = pd.DataFrame(all_rows)
    out = RESULTS_DIR / "reset_bench.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved to: {out}")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    summary = (
        df.groupby("geometry_mode")["reset_time_s"]
        .agg(mean_ms=lambda x: x.mean() * 1000,
             median_ms=lambda x: x.median() * 1000,
             min_ms=lambda x: x.min() * 1000,
             max_ms=lambda x: x.max() * 1000)
        .reset_index()
    )
    print(summary.to_string(index=False))

    if "box" in summary["geometry_mode"].values and "vhacd" in summary["geometry_mode"].values:
        box_t = summary.loc[summary["geometry_mode"] == "box", "mean_ms"].iloc[0]
        vhacd_t = summary.loc[summary["geometry_mode"] == "vhacd", "mean_ms"].iloc[0]
        print(f"\nReset overhead (vhacd / box): {vhacd_t / box_t:.2f}×  ({vhacd_t - box_t:.1f}ms extra per reset)")
        print(f"Cost per 1M resets: box={box_t*1e6/3600:.0f} GPU-hours  vhacd={vhacd_t*1e6/3600:.0f} GPU-hours  (wall-clock at 1 reset/ms scale)")


if __name__ == "__main__":
    main()

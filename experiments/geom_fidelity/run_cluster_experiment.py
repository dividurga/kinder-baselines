"""Gate + clutter experiment: task-conditioned geometry fidelity.

Three conditions, same random seeds:
  all_vhacd  — gate=vhacd, clutter=vhacd   (accurate, slow baseline)
  all_box    — gate=box,   clutter=box     (coarse everywhere, may fail at gate)
  mixed      — gate=vhacd, clutter=box     (thesis: VHACD where it matters, box elsewhere)

Expected result:
  success:  all_vhacd ≈ mixed  >>  all_box   (gate geometry decides path existence)
  speed:    mixed  <  all_vhacd              (fewer VHACD hulls to check)

Use --calibrate to sweep gate separations and find the gap where box starts failing.

Usage:
    python run_cluster_experiment.py
    python run_cluster_experiment.py --n_trials 50 --gate_sep 0.17
    python run_cluster_experiment.py --calibrate --gate_seps 0.14 0.16 0.18 0.20
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pybullet_helpers.motion_planning as _pbmp
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.motion_planning import (
    MotionPlanningHyperparameters,
    run_motion_planning,
)

from kinder.envs.kinematic3d.cluster_reach3d import (
    ClusterReach3DConfig,
    ObjectCentricClusterReach3DEnv,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = Path(__file__).parent / "assets"

CONDITIONS = {
    "all_vhacd": dict(gate_mode="vhacd", clutter_mode="vhacd"),
    "all_box":   dict(gate_mode="box",   clutter_mode="box"),
    "mixed":     dict(gate_mode="vhacd", clutter_mode="box"),
}


# ---------------------------------------------------------------------------
# Collision-check instrumentation (same pattern as run_experiment.py)
# ---------------------------------------------------------------------------

@contextmanager
def instrumented_collision_fn(
    counter: list[int], timer: list[float]
) -> Generator[None, None, None]:
    import pybullet_helpers.inverse_kinematics as _pbi
    original = _pbi.check_collisions_with_held_object
    def wrapped(*args, **kwargs):
        counter[0] += 1
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        timer[0] += time.perf_counter() - t0
        return result
    _pbi.check_collisions_with_held_object = wrapped
    _orig_mp = _pbmp.check_collisions_with_held_object
    _pbmp.check_collisions_with_held_object = wrapped
    try:
        yield
    finally:
        _pbi.check_collisions_with_held_object = original
        _pbmp.check_collisions_with_held_object = _orig_mp


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    env: ObjectCentricClusterReach3DEnv,
    trial_seed: int,
    hyperparams: MotionPlanningHyperparameters,
    rng: np.random.Generator,
) -> dict:
    env.reset(seed=trial_seed)
    q_start = list(env.robot.arm.get_joint_positions())

    target_pos = env._get_obs().target_position
    u1, u2, u3 = rng.random(size=3)
    quat = (
        float(np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)),
        float(np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)),
        float(np.sqrt(u1) * np.sin(2 * np.pi * u3)),
        float(np.sqrt(u1) * np.cos(2 * np.pi * u3)),
    )
    try:
        q_goal = inverse_kinematics(
            env.robot.arm, Pose(target_pos, quat), validate=True, set_joints=False
        )
        ik_success = True
    except InverseKinematicsError:
        return {
            "ik_success": False, "success": False,
            "planning_time_s": 0.0,
            "n_collision_checks": 0, "collision_check_time_s": 0.0,
        }

    counter, timer = [0], [0.0]
    with instrumented_collision_fn(counter, timer):
        t0 = time.perf_counter()
        plan = run_motion_planning(
            env.robot.arm,
            initial_positions=q_start,
            target_positions=list(q_goal),
            collision_bodies=set(env.get_obstacle_ids()),
            seed=trial_seed,
            physics_client_id=env.physics_client_id,
            hyperparameters=hyperparams,
        )
        planning_time = time.perf_counter() - t0

    return {
        "ik_success": ik_success,
        "success": plan is not None,
        "planning_time_s": planning_time,
        "n_collision_checks": counter[0],
        "collision_check_time_s": timer[0],
    }


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    conditions: dict[str, dict],
    gate_sep: float,
    n_trials: int,
    base_seed: int,
    birrt_num_attempts: int,
    birrt_num_iters: int,
    n_clutter: int,
) -> pd.DataFrame:
    hyperparams = MotionPlanningHyperparameters(
        birrt_num_attempts=birrt_num_attempts,
        birrt_num_iters=birrt_num_iters,
    )
    all_results = []

    for cond_name, modes in conditions.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}  gate_sep={gate_sep:.3f}m  ({n_trials} trials)")
        print(f"  gate={modes['gate_mode']}  clutter={modes['clutter_mode']}")
        print(f"{'='*60}")

        config = ClusterReach3DConfig(
            gate_mode=modes["gate_mode"],
            clutter_mode=modes["clutter_mode"],
            gate_sep_m=gate_sep,
            n_clutter=n_clutter,
            assets_dir=ASSETS_DIR,
        )
        env = ObjectCentricClusterReach3DEnv(config=config)
        rng = np.random.default_rng(base_seed)

        successes = ik_failures = 0
        for trial in range(n_trials):
            result = run_trial(env, base_seed + trial, hyperparams, rng)
            result.update({
                "condition": cond_name,
                "gate_mode": modes["gate_mode"],
                "clutter_mode": modes["clutter_mode"],
                "gate_sep_m": gate_sep,
                "trial": trial,
            })
            all_results.append(result)

            if not result["ik_success"]:
                ik_failures += 1
                status = "IK fail"
            elif result["success"]:
                successes += 1
                status = "OK     "
            else:
                status = "no plan"

            print(
                f"  [{trial+1:3d}/{n_trials}] {status}  "
                f"plan={result['planning_time_s']:.3f}s  "
                f"checks={result['n_collision_checks']:5d}  "
                f"check_t={result['collision_check_time_s']*1000:.2f}ms"
            )

        valid = n_trials - ik_failures
        sr = successes / valid if valid > 0 else float("nan")
        print(f"\n  success_rate={sr:.1%}  ik_failures={ik_failures}/{n_trials}")

    return pd.DataFrame(all_results)


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    valid = df[df["ik_success"]]
    summary = (
        valid.groupby("condition")
        .agg(
            success_rate=("success", "mean"),
            mean_planning_time_s=("planning_time_s", "mean"),
            mean_n_checks=("n_collision_checks", "mean"),
            mean_check_time_ms=("collision_check_time_s", lambda x: x.mean() * 1000),
            n_trials=("success", "count"),
        )
        .reset_index()
    )
    # Print in a fixed order.
    order = ["all_vhacd", "mixed", "all_box"]
    summary["_order"] = summary["condition"].map({c: i for i, c in enumerate(order)})
    summary = summary.sort_values("_order").drop(columns="_order")
    print(summary.to_string(index=False))

    for cond in ["all_box", "mixed"]:
        if cond in summary["condition"].values and "all_vhacd" in summary["condition"].values:
            vhacd_t = summary.loc[summary["condition"] == "all_vhacd", "mean_planning_time_s"].iloc[0]
            cond_t  = summary.loc[summary["condition"] == cond,        "mean_planning_time_s"].iloc[0]
            vhacd_sr = summary.loc[summary["condition"] == "all_vhacd", "success_rate"].iloc[0]
            cond_sr  = summary.loc[summary["condition"] == cond,        "success_rate"].iloc[0]
            print(f"\n{cond} vs all_vhacd:"
                  f"  speedup={vhacd_t/cond_t:.2f}×"
                  f"  success_diff={cond_sr - vhacd_sr:+.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--gate_sep", type=float, default=0.17,
                        help="Gate center-to-center Y separation in metres")
    parser.add_argument("--n_clutter", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--birrt_num_attempts", type=int, default=10)
    parser.add_argument("--birrt_num_iters", type=int, default=200)
    parser.add_argument("--conditions", nargs="+",
                        default=list(CONDITIONS.keys()),
                        choices=list(CONDITIONS.keys()),
                        help="Subset of conditions to run")
    # Calibration sweep: find the gate_sep where box starts failing.
    parser.add_argument("--calibrate", action="store_true",
                        help="Sweep gate separations to find the critical gap")
    parser.add_argument("--gate_seps", nargs="+", type=float,
                        default=[0.13, 0.15, 0.17, 0.19, 0.21],
                        help="Separations to sweep in calibration mode")
    args = parser.parse_args()

    selected_conditions = {k: CONDITIONS[k] for k in args.conditions}

    if args.calibrate:
        print("=== CALIBRATION SWEEP ===")
        cal_rows = []
        for sep in args.gate_seps:
            print(f"\n--- gate_sep={sep:.2f}m ---")
            df = run_experiment(
                selected_conditions, sep,
                n_trials=20,  # quick sweep
                base_seed=args.seed,
                birrt_num_attempts=args.birrt_num_attempts,
                birrt_num_iters=args.birrt_num_iters,
                n_clutter=args.n_clutter,
            )
            df["gate_sep_sweep"] = sep
            cal_rows.append(df)
        df_all = pd.concat(cal_rows, ignore_index=True)
        out = RESULTS_DIR / "cluster_calibration.csv"
        df_all.to_csv(out, index=False)
        print(f"\nCalibration results → {out}")

        # Print pivot: success_rate by (condition, gate_sep).
        valid = df_all[df_all["ik_success"]]
        pivot = valid.pivot_table(
            values="success", index="gate_sep_sweep", columns="condition", aggfunc="mean"
        )
        print("\nSuccess rate by gate separation:")
        print(pivot.to_string())
    else:
        df = run_experiment(
            selected_conditions, args.gate_sep,
            n_trials=args.n_trials,
            base_seed=args.seed,
            birrt_num_attempts=args.birrt_num_attempts,
            birrt_num_iters=args.birrt_num_iters,
            n_clutter=args.n_clutter,
        )
        out = RESULTS_DIR / "cluster_results.csv"
        df.to_csv(out, index=False)
        print(f"\nResults → {out}")
        print_summary(df)

"""Row experiment: 6 mugs in a fixed fence, EE must reach the gap between mug 3 and 4.

Compares box (AABB) vs vhacd collision geometry.
Use --visualize to open a GUI replay of the first successful plan per mode.
Use --mug_spacing to sweep gap widths.

Usage:
    python run_row_experiment.py --n_trials 50
    python run_row_experiment.py --n_trials 5 --visualize
    python run_row_experiment.py --sweep_spacing
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pybullet as p
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

from kinder.envs.kinematic3d.row_reach3d import (
    RowReach3DConfig,
    ObjectCentricRowReach3DEnv,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = Path(__file__).parent / "assets"


@contextmanager
def instrumented_collision_fn(counter: list[int], timer: list[float]) -> Generator:
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


def run_trial(
    env: ObjectCentricRowReach3DEnv,
    trial_seed: int,
    hyperparams: MotionPlanningHyperparameters,
) -> tuple[dict, list[JointPositions] | None]:
    env.reset(seed=trial_seed)
    q_start = list(env.robot.arm.get_joint_positions())
    target_pos = env._get_obs().target_position
    rng = np.random.default_rng(trial_seed)
    u1, u2, u3 = rng.random(size=3)
    quat = (
        float(np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)),
        float(np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)),
        float(np.sqrt(u1) * np.sin(2 * np.pi * u3)),
        float(np.sqrt(u1) * np.cos(2 * np.pi * u3)),
    )
    target_pose = Pose(target_pos, quat)

    try:
        q_goal = inverse_kinematics(env.robot.arm, target_pose, validate=True, set_joints=False)
        ik_success = True
    except InverseKinematicsError:
        return {"ik_success": False, "success": False,
                "planning_time_s": 0.0, "n_collision_checks": 0,
                "collision_check_time_s": 0.0}, None

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
    }, plan


def visualize_plan(
    mode: str,
    trial_seed: int,
    plan: list[JointPositions],
    mug_spacing: float,
    target_overrides: dict | None = None,
) -> None:
    print(f"\n  [viz] Opening GUI for mode={mode} seed={trial_seed} ...")
    config = RowReach3DConfig(
        geometry_mode=mode, mug_spacing=mug_spacing, assets_dir=ASSETS_DIR,
        **(target_overrides or {}),
    )
    gui_env = ObjectCentricRowReach3DEnv(config=config, use_gui=True)
    gui_env.reset(seed=trial_seed)
    pcid = gui_env.physics_client_id

    ee_positions = []
    for q in plan:
        gui_env.robot.arm.set_joints(q)
        p.stepSimulation(physicsClientId=pcid)
        ee_positions.append(gui_env.robot.arm.get_end_effector_pose().position)
        time.sleep(0.04)

    for i in range(len(ee_positions) - 1):
        p.addUserDebugLine(
            ee_positions[i], ee_positions[i + 1],
            lineColorRGB=[0.0, 0.8, 0.2], lineWidth=3, physicsClientId=pcid,
        )

    # Mark mug positions with Y labels so user can see the row.
    ys = gui_env.mug_y_positions()
    for idx, y in enumerate(ys):
        p.addUserDebugText(
            f"M{idx+1}", (config.row_x, y, config.row_z + 0.08),
            textColorRGB=[1.0, 0.8, 0.0], textSize=1.2, physicsClientId=pcid,
        )
    from pybullet_helpers.geometry import get_pose
    target_pos = get_pose(gui_env.target_id, pcid).position
    p.addUserDebugText(
        "TARGET", target_pos,
        textColorRGB=[0.2, 1.0, 0.2], textSize=1.4, physicsClientId=pcid,
    )
    p.addUserDebugText(
        "START", ee_positions[0],
        textColorRGB=[0.2, 0.4, 1.0], textSize=1.4, physicsClientId=pcid,
    )

    print(f"  [viz] {len(plan)} steps. Press Enter to close.")
    input()
    p.disconnect(pcid)


def run_experiment(
    modes: list[str],
    mug_spacing: float,
    n_trials: int,
    base_seed: int,
    birrt_num_attempts: int,
    birrt_num_iters: int,
    visualize: bool,
    target_overrides: dict | None = None,
) -> pd.DataFrame:
    target_overrides = target_overrides or {}
    hyperparams = MotionPlanningHyperparameters(
        birrt_num_attempts=birrt_num_attempts,
        birrt_num_iters=birrt_num_iters,
    )
    all_results = []
    first_successes: dict[str, tuple[int, list]] = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"mode={mode}  spacing={mug_spacing:.3f}m  ({n_trials} trials)")
        print(f"{'='*60}")
        config = RowReach3DConfig(
            geometry_mode=mode, mug_spacing=mug_spacing, assets_dir=ASSETS_DIR,
            **target_overrides,
        )
        env = ObjectCentricRowReach3DEnv(config=config)

        successes = ik_failures = 0
        for trial in range(n_trials):
            result, plan = run_trial(env, base_seed + trial, hyperparams)
            result.update({"geometry_mode": mode, "trial": trial, "mug_spacing": mug_spacing})
            all_results.append(result)

            if not result["ik_success"]:
                ik_failures += 1
                status = "IK fail"
            elif result["success"]:
                successes += 1
                status = "OK     "
                if visualize and mode not in first_successes and plan is not None:
                    first_successes[mode] = (base_seed + trial, plan)
            else:
                status = "no plan"

            print(
                f"  [{trial+1:3d}/{n_trials}] {status}  "
                f"plan={result['planning_time_s']:.3f}s  "
                f"checks={result['n_collision_checks']:5d}  "
                f"check_t={result['collision_check_time_s']*1000:.1f}ms"
            )

        valid = n_trials - ik_failures
        sr = successes / valid if valid > 0 else float("nan")
        print(f"\n  success_rate={sr:.1%}  ik_failures={ik_failures}/{n_trials}")

    if visualize:
        for mode, (seed, plan) in first_successes.items():
            visualize_plan(mode, seed, plan, mug_spacing, target_overrides)

    return pd.DataFrame(all_results)


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    valid = df[df["ik_success"]]
    summary = (
        valid.groupby("geometry_mode")
        .agg(
            success_rate=("success", "mean"),
            mean_planning_time_s=("planning_time_s", "mean"),
            mean_n_checks=("n_collision_checks", "mean"),
            mean_check_time_ms=("collision_check_time_s", lambda x: x.mean() * 1000),
            n_trials=("success", "count"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    if "box" in summary["geometry_mode"].values and "vhacd" in summary["geometry_mode"].values:
        box_t  = summary.loc[summary["geometry_mode"] == "box",   "mean_planning_time_s"].iloc[0]
        vhcd_t = summary.loc[summary["geometry_mode"] == "vhacd", "mean_planning_time_s"].iloc[0]
        box_sr = summary.loc[summary["geometry_mode"] == "box",   "success_rate"].iloc[0]
        vh_sr  = summary.loc[summary["geometry_mode"] == "vhacd", "success_rate"].iloc[0]
        print(f"\nSpeedup box vs vhacd: {vhcd_t/box_t:.2f}×")
        print(f"Success: box={box_sr:.1%}  vhacd={vh_sr:.1%}  diff={vh_sr-box_sr:+.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--mug_spacing", type=float, default=0.13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--birrt_num_attempts", type=int, default=10)
    parser.add_argument("--birrt_num_iters", type=int, default=200)
    parser.add_argument("--geometry_modes", nargs="+", default=["box", "vhacd"])
    parser.add_argument("--visualize", action="store_true",
                        help="Open GUI replay of first successful plan per mode")
    parser.add_argument("--gap_target", action="store_true",
                        help="Exp 5 mode: lock target into the mug 3-4 gap (tight passage demo)")
    parser.add_argument("--sweep_spacing", action="store_true",
                        help="Quick sweep over mug spacings to find the right gap")
    parser.add_argument("--spacings", nargs="+", type=float,
                        default=[0.11, 0.13, 0.15, 0.17])
    args = parser.parse_args()

    # Compute target zone for gap_target mode: lock to mug 3-4 gap.
    target_overrides: dict = {}
    if args.gap_target:
        y3 = 0.15 + 2 * args.mug_spacing   # mug 3 centre
        y4 = 0.15 + 3 * args.mug_spacing   # mug 4 centre
        gap_y = (y3 + y4) / 2.0
        margin = 0.015
        target_overrides = dict(
            target_x_lb=0.38, target_x_ub=0.45,
            target_y_lb=gap_y - margin, target_y_ub=gap_y + margin,
            target_z_lb=0.48, target_z_ub=0.56,
        )
        print(f"Gap target zone: y=[{target_overrides['target_y_lb']:.3f}, {target_overrides['target_y_ub']:.3f}]")

    if args.sweep_spacing:
        rows = []
        for sp in args.spacings:
            df = run_experiment(
                args.geometry_modes, sp,
                n_trials=20, base_seed=args.seed,
                birrt_num_attempts=args.birrt_num_attempts,
                birrt_num_iters=args.birrt_num_iters,
                visualize=False,
            )
            rows.append(df)
        df_all = pd.concat(rows, ignore_index=True)
        valid = df_all[df_all["ik_success"]]
        pivot = valid.pivot_table(
            values="success", index="mug_spacing", columns="geometry_mode", aggfunc="mean"
        )
        time_pivot = valid.pivot_table(
            values="planning_time_s", index="mug_spacing", columns="geometry_mode", aggfunc="mean"
        )
        print("\nSuccess rate by spacing:")
        print(pivot.to_string())
        print("\nMean planning time (s) by spacing:")
        print(time_pivot.to_string())
        df_all.to_csv(RESULTS_DIR / "row_sweep.csv", index=False)
    else:
        df = run_experiment(
            args.geometry_modes, args.mug_spacing,
            n_trials=args.n_trials, base_seed=args.seed,
            birrt_num_attempts=args.birrt_num_attempts,
            birrt_num_iters=args.birrt_num_iters,
            visualize=args.visualize,
            target_overrides=target_overrides,
        )
        df.to_csv(RESULTS_DIR / "row_results.csv", index=False)
        print_summary(df)

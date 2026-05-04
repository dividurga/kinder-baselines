"""Geometry fidelity experiment: AABB (box) vs VHACD collision geometry.

For each geometry mode, runs N trials of a reach task where TidyBot-Kinova
must plan an arm trajectory around YCB obstacles. Uses BiRRT (via
pybullet_helpers.motion_planning.run_motion_planning).

Metrics recorded per trial:
  - success: bool (plan found)
  - planning_time_s: wall-clock seconds for BiRRT call
  - n_collision_checks: number of collision function evaluations
  - collision_check_time_s: cumulative time inside collision checking
  - ik_success: bool (IK found a valid goal config)

Usage:
    python run_experiment.py
    python run_experiment.py --n_trials 10 --n_obstacles 3 --seed 0
    python run_experiment.py --geometry_modes box vhacd --visualize
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

from kinder.envs.kinematic3d.geom_fidelity_reach3d import (
    GeomFidelityReach3DConfig,
    ObjectCentricGeomFidelityReach3DEnv,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ASSETS_DIR = Path(__file__).parent / "assets"

DEFAULT_OBSTACLE_NAMES = ("025_mug",)  # all mugs for sanity check


# ---------------------------------------------------------------------------
# Collision-check instrumentation
# ---------------------------------------------------------------------------

@contextmanager
def instrumented_collision_fn(
    counter: list[int], timer: list[float]
) -> Generator[None, None, None]:
    """Monkey-patch check_collisions_with_held_object to count calls and time."""
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
    env: ObjectCentricGeomFidelityReach3DEnv,
    trial_seed: int,
    hyperparams: MotionPlanningHyperparameters,
    rng: np.random.Generator,
) -> tuple[dict, list[JointPositions] | None]:
    """Run one reach trial. Returns (result_dict, plan_or_None)."""
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
    target_pose = Pose(target_pos, quat)

    try:
        q_goal = inverse_kinematics(
            env.robot.arm, target_pose, validate=True, set_joints=False
        )
        ik_success = True
    except InverseKinematicsError:
        return {
            "ik_success": False,
            "success": False,
            "planning_time_s": 0.0,
            "n_collision_checks": 0,
            "collision_check_time_s": 0.0,
        }, None

    obstacle_ids = set(env.get_obstacle_ids())

    counter = [0]
    timer = [0.0]
    with instrumented_collision_fn(counter, timer):
        t0 = time.perf_counter()
        plan = run_motion_planning(
            env.robot.arm,
            initial_positions=q_start,
            target_positions=list(q_goal),
            collision_bodies=obstacle_ids,
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


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_plan(
    mode: str,
    trial_seed: int,
    plan: list[JointPositions],
    n_obstacles: int,
    obstacle_names: tuple[str, ...],
    step_sleep: float = 0.04,
) -> None:
    """Replay a plan in a GUI window, drawing the EE path as a line."""
    print(f"\n  [viz] Opening GUI for mode={mode} seed={trial_seed} ...")

    config = GeomFidelityReach3DConfig(
        geometry_mode=mode,
        num_obstacles=n_obstacles,
        obstacle_names=obstacle_names,
        assets_dir=ASSETS_DIR,
    )
    gui_env = ObjectCentricGeomFidelityReach3DEnv(config=config, use_gui=True)
    gui_env.reset(seed=trial_seed)
    pcid = gui_env.physics_client_id

    # Step through the plan, recording EE positions.
    ee_positions: list[tuple] = []
    for q in plan:
        gui_env.robot.arm.set_joints(q)
        p.stepSimulation(physicsClientId=pcid)
        ee_pos = gui_env.robot.arm.get_end_effector_pose().position
        ee_positions.append(ee_pos)
        time.sleep(step_sleep)

    # Draw the EE trajectory as connected line segments.
    for i in range(len(ee_positions) - 1):
        p.addUserDebugLine(
            ee_positions[i],
            ee_positions[i + 1],
            lineColorRGB=[0.0, 0.8, 0.2],
            lineWidth=3,
            physicsClientId=pcid,
        )

    # Mark start (blue) and end (red) with debug spheres.
    p.addUserDebugText(
        "START", ee_positions[0], textColorRGB=[0.2, 0.4, 1.0],
        textSize=1.5, physicsClientId=pcid,
    )
    p.addUserDebugText(
        "GOAL", ee_positions[-1], textColorRGB=[1.0, 0.2, 0.2],
        textSize=1.5, physicsClientId=pcid,
    )

    print(f"  [viz] Plan replayed ({len(plan)} steps). Press Enter to close.")
    input()
    p.disconnect(pcid)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    geometry_modes: list[str],
    n_trials: int,
    n_obstacles: int,
    obstacle_names: tuple[str, ...],
    base_seed: int,
    birrt_num_attempts: int,
    birrt_num_iters: int,
    visualize: bool,
) -> pd.DataFrame:
    hyperparams = MotionPlanningHyperparameters(
        birrt_num_attempts=birrt_num_attempts,
        birrt_num_iters=birrt_num_iters,
    )

    all_results = []
    # (mode, seed, plan) for first success per mode — used for visualisation.
    first_successes: dict[str, tuple[int, list[JointPositions]]] = {}

    for mode in geometry_modes:
        print(f"\n{'='*60}")
        print(f"Geometry mode: {mode}  ({n_trials} trials, {n_obstacles} obstacles)")
        print(f"{'='*60}")

        config = GeomFidelityReach3DConfig(
            geometry_mode=mode,
            num_obstacles=n_obstacles,
            obstacle_names=obstacle_names,
            assets_dir=ASSETS_DIR,
        )
        env = ObjectCentricGeomFidelityReach3DEnv(config=config)
        rng = np.random.default_rng(base_seed)

        successes = 0
        ik_failures = 0
        for trial in range(n_trials):
            trial_seed = base_seed + trial
            result, plan = run_trial(env, trial_seed, hyperparams, rng)

            result.update({
                "geometry_mode": mode,
                "trial": trial,
                "n_obstacles": n_obstacles,
            })
            all_results.append(result)

            if not result["ik_success"]:
                ik_failures += 1
                status = "IK fail"
            elif result["success"]:
                successes += 1
                status = "OK     "
                if visualize and mode not in first_successes and plan is not None:
                    first_successes[mode] = (trial_seed, plan)
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

    # Visualise first success for each mode.
    if visualize:
        for mode, (seed, plan) in first_successes.items():
            visualize_plan(mode, seed, plan, n_obstacles, obstacle_names)

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
        box_t = summary.loc[summary["geometry_mode"] == "box", "mean_planning_time_s"].iloc[0]
        vhacd_t = summary.loc[summary["geometry_mode"] == "vhacd", "mean_planning_time_s"].iloc[0]
        box_sr = summary.loc[summary["geometry_mode"] == "box", "success_rate"].iloc[0]
        vhacd_sr = summary.loc[summary["geometry_mode"] == "vhacd", "success_rate"].iloc[0]
        print(f"\nSpeedup (vhacd/box planning time): {vhacd_t/box_t:.2f}×")
        print(f"Success rate: box={box_sr:.1%}  vhacd={vhacd_sr:.1%}  diff={vhacd_sr-box_sr:+.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_modes", nargs="+", default=["box", "vhacd"])
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--n_obstacles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--birrt_num_attempts", type=int, default=10)
    parser.add_argument("--birrt_num_iters", type=int, default=200)
    parser.add_argument(
        "--obstacle_names", nargs="+", default=list(DEFAULT_OBSTACLE_NAMES)
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="After all trials, replay the first successful plan per mode in a GUI window.",
    )
    args = parser.parse_args()

    df = run_experiment(
        geometry_modes=args.geometry_modes,
        n_trials=args.n_trials,
        n_obstacles=args.n_obstacles,
        obstacle_names=tuple(args.obstacle_names),
        base_seed=args.seed,
        birrt_num_attempts=args.birrt_num_attempts,
        birrt_num_iters=args.birrt_num_iters,
        visualize=args.visualize,
    )

    out_path = RESULTS_DIR / "geom_fidelity_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    print_summary(df)

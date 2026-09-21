"""Headless packaging check; this is not a policy or scoring evaluation."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    import leisaac.tasks
    from isaaclab_tasks.utils import parse_env_cfg

    repo = Path(__file__).resolve().parents[1]
    assert Path(leisaac.tasks.__file__).resolve().is_relative_to(repo), "Wrong leisaac imported"
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.use_teleop_device("xtrainerleader")
    cfg.seed = 123
    cfg.enable_visualization = False
    # Keep dynamics and task checks; remove image observations for this smoke check.
    for name in ("left_wrist", "right_wrist", "top", "stereo_left", "stereo_right"):
        setattr(cfg.scene, name, None)
        if hasattr(cfg.observations.policy, name):
            setattr(cfg.observations.policy, name, None)
    env = gym.make(args.task, cfg=cfg)
    try:
        with torch.inference_mode():
            env.reset()
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            for _ in range(3):
                env.step(action)
        print("SMOKE_RESULT=" + json.dumps({
            "task": args.task, "status": "passed",
            "reset": True, "steps": 3, "action_shape": list(action.shape),
            "policy_evaluated": False, "cameras_enabled": False,
        }), flush=True)
    finally:
        env.close()
finally:
    app.close()

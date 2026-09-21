"""Script to run a leisaac inference with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
import json
import os

_print = print


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return _print(*args, **kwargs)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac inference for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument("--eval_rounds", type=int, default=0, help="Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual reset.")
parser.add_argument("--policy_type", type=str, default="gr00tn1.5", help="Type of policy to use. support gr00tn1.5, lerobot-<model_type>, openpi, xtrainer_act.")
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")

"""
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-XTrainer-PickCube-v0 \
    --eval_rounds=10 \
    --policy_type=xtrainer_act \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Grab cube and place into plate" \
    --device=cuda \
    --enable_cameras \
    --policy_checkpoint_path="/home/wyt/yc/lerobot/outputs/train/act_xtrainer_lift_cube/checkpoints/last/pretrained_model"
"""

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import time
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401
from leisaac.utils.env_utils import get_task_type, dynamic_reset_gripper_effort_limit_sim

import carb
import omni

import leisaac.tasks


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True


def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    """Preprocess the observation dictionary to the format expected by the policy."""
    if model_type in ["gr00tn1.5", "lerobot", "openpi", "xtrainer_act"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    else:
        raise ValueError(f"Model type {model_type} not supported")


def print_task3_timeout_diagnostics(env):
    """Print object placement and rest-pose status when Task3 times out."""
    if args_cli.task != "task3":
        return

    try:
        from leisaac.tasks.task3.xtrainer_pickup_recognition_env_cfg import (
            TASK3_SUCCESS_CONTAINER_NAME,
            TASK3_SUCCESS_HEIGHT_RANGE,
            TASK3_SUCCESS_OBJECT_NAMES,
            TASK3_SUCCESS_OBJECT_OFFSETS,
            TASK3_SUCCESS_OBJECT_REGIONS,
            TASK3_SUCCESS_OBJECT_REGION_MAP,
            TASK3_SUCCESS_OFFSET_FRAME,
        )
        from leisaac.tasks.task3.mdp.terminations import _get_scene_entity_pos_w, get_detection_point_pos_w
        from leisaac.utils.robot_utils import is_xtrainer_at_rest_pose
    except Exception as exc:
        print(f"[Task3Diag] Unable to import diagnostics: {exc}")
        return

    try:
        container_pos_w = _get_scene_entity_pos_w(env, TASK3_SUCCESS_CONTAINER_NAME)
        container_local = container_pos_w - env.scene.env_origins
        container_xyz = container_local[0, :3].detach().cpu().tolist()
        print(
            "[Task3Diag] container "
            f"{TASK3_SUCCESS_CONTAINER_NAME} local_xyz="
            f"({container_xyz[0]:.4f}, {container_xyz[1]:.4f}, {container_xyz[2]:.4f})"
        )

        use_offset_frame = "local" if TASK3_SUCCESS_OFFSET_FRAME == "local" else "world"
        all_in_region = True
        for object_name in TASK3_SUCCESS_OBJECT_NAMES:
            object_entity = env.scene[object_name]
            object_offset = TASK3_SUCCESS_OBJECT_OFFSETS.get(object_name, (0.0, 0.0, 0.0))
            object_pos_w = get_detection_point_pos_w(
                env=env,
                object_entity=object_entity,
                object_offset=object_offset,
                offset_frame=use_offset_frame,
                apply_object_scale=True,
                use_visual_center_anchor=True,
            )
            object_local = object_pos_w - env.scene.env_origins
            obj = object_local[0, :3].detach().cpu()
            ctr = container_local[0, :3].detach().cpu()
            rel = obj - ctr

            region_name = TASK3_SUCCESS_OBJECT_REGION_MAP.get(object_name)
            region_cfg = TASK3_SUCCESS_OBJECT_REGIONS.get(region_name, {})
            x_range = region_cfg.get("x_range")
            y_range = region_cfg.get("y_range")
            h_range = region_cfg.get("height_range", TASK3_SUCCESS_HEIGHT_RANGE)
            in_x = bool(x_range[0] < rel[0].item() < x_range[1])
            in_y = bool(y_range[0] < rel[1].item() < y_range[1])
            in_z = bool(h_range[0] < rel[2].item() < h_range[1])
            in_region = in_x and in_y and in_z
            all_in_region = all_in_region and in_region
            print(
                "[Task3Diag] "
                f"{object_name} -> {region_name}: "
                f"rel=({rel[0].item():.4f}, {rel[1].item():.4f}, {rel[2].item():.4f}) "
                f"x_ok={in_x} y_ok={in_y} z_ok={in_z} in_region={in_region}"
            )

        joint_pos = env.scene["robot"].data.joint_pos
        joint_names = env.scene["robot"].data.joint_names
        rest = bool(is_xtrainer_at_rest_pose(joint_pos, joint_names)[0].item())
        print(f"[Task3Diag] all_objects_in_region={all_in_region} robot_rest_pose={rest}")
    except Exception as exc:
        print(f"[Task3Diag] Failed to compute diagnostics: {exc}")


def _tensor_xyz(value):
    return [float(x) for x in value.detach().cpu().reshape(-1)[:3].tolist()]


_TASK3_ACTION_JOINT_NAMES = [
    "J1_1", "J1_2", "J1_3", "J1_4", "J1_5", "J1_6", "J1_7", "J1_8",
    "J2_1", "J2_2", "J2_3", "J2_4", "J2_5", "J2_6", "J2_7", "J2_8",
]


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _get_task3_action_joint_pos(env):
    try:
        joint_names = list(env.scene["robot"].data.joint_names)
        joint_index = {name: idx for idx, name in enumerate(joint_names)}
        indices = [joint_index[name] for name in _TASK3_ACTION_JOINT_NAMES]
        return env.scene["robot"].data.joint_pos[:, indices]
    except Exception:
        return None


def maybe_apply_task3_first_grasp_protection(env, action, rollout_step: int):
    """Lightly slow the early Task3 approach motion without changing gripper logic."""
    if args_cli.task != "task3":
        return action

    protect_steps = _get_env_int("XTRAINER_TASK3_FIRST_GRASP_PROTECT_STEPS", 0)
    if protect_steps <= 0 or rollout_step >= protect_steps:
        return action

    arm_scale = _get_env_float("XTRAINER_TASK3_FIRST_GRASP_ARM_SCALE", 0.75)
    if arm_scale >= 0.999:
        return action

    current_joint_pos = _get_task3_action_joint_pos(env)
    if current_joint_pos is None:
        return action

    try:
        protected = action.clone()
        if protected.shape != current_joint_pos.shape:
            return action
        arm_indices = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
        protected[:, arm_indices] = current_joint_pos[:, arm_indices] + arm_scale * (
            protected[:, arm_indices] - current_joint_pos[:, arm_indices]
        )
        return protected
    except Exception:
        return action


def _get_task3_object_positions(env):
    from leisaac.tasks.task3.xtrainer_pickup_recognition_env_cfg import (
        TASK3_SUCCESS_OBJECT_NAMES,
        TASK3_SUCCESS_OBJECT_OFFSETS,
        TASK3_SUCCESS_OFFSET_FRAME,
    )
    from leisaac.tasks.task3.mdp.terminations import get_detection_point_pos_w

    use_offset_frame = "local" if TASK3_SUCCESS_OFFSET_FRAME == "local" else "world"
    objects = {}
    for object_name in TASK3_SUCCESS_OBJECT_NAMES:
        object_entity = env.scene[object_name]
        object_offset = TASK3_SUCCESS_OBJECT_OFFSETS.get(object_name, (0.0, 0.0, 0.0))
        object_pos_w = get_detection_point_pos_w(
            env=env,
            object_entity=object_entity,
            object_offset=object_offset,
            offset_frame=use_offset_frame,
            apply_object_scale=True,
            use_visual_center_anchor=True,
        )
        object_local = object_pos_w - env.scene.env_origins
        objects[object_name] = _tensor_xyz(object_local[0, :3])
    return objects


def _get_task3_eef_positions(env):
    eef = {}
    for scene_key, out_key in (("left_ee_frame", "left"), ("right_ee_frame", "right")):
        try:
            sensor = env.scene[scene_key]
            pos_w = sensor.data.target_pos_w[:, 0, :] if sensor.data.target_pos_w.ndim == 3 else sensor.data.target_pos_w
            pos_local = pos_w - env.scene.env_origins
            eef[out_key] = _tensor_xyz(pos_local[0, :3])
        except Exception:
            eef[out_key] = None
    return eef


def append_task3_trajectory_log(env, step_index: int, action=None, source: str = "policy"):
    """Append Task3 EEF/object positions for rollout-vs-demo alignment debugging."""
    log_path = os.environ.get("XTRAINER_TASK3_TRAJ_LOG")
    if args_cli.task != "task3" or not log_path:
        return

    try:
        objects = _get_task3_object_positions(env)
        eef = _get_task3_eef_positions(env)
        joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().tolist()
        action_values = None
        if action is not None:
            action_values = action.detach().cpu().reshape(-1).tolist()
        record = {
            "source": source,
            "step": int(step_index),
            "eef": eef,
            "objects": objects,
            "joint_pos": [float(x) for x in joint_pos],
            "action": None if action_values is None else [float(x) for x in action_values],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        if step_index == 0:
            print(f"[Task3Traj] Failed to append trajectory log: {exc}")


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s

    # modify configuration
    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # create policy
    model_type = args_cli.policy_type
    if args_cli.policy_type == "gr00tn1.5":
        from leisaac.policy import Gr00tServicePolicyClient
        from isaaclab.sensors import Camera

        if task_type == "so101leader":
            modality_keys = ["single_arm", "gripper"]
        else:
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")

        policy = Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        )
    elif "lerobot" in args_cli.policy_type:
        from leisaac.policy import LeRobotServicePolicyClient
        from isaaclab.sensors import Camera

        model_type = 'lerobot'

        policy_type = args_cli.policy_type.split("-")[1]
        camera_infos = {
            key: sensor.image_shape
            for key, sensor in env.scene.sensors.items()
            if isinstance(sensor, Camera)
        }
        if task_type == "xtrainerleader":
            xtrainer_policy_cameras = {"top", "left_wrist", "right_wrist"}
            camera_infos = {
                key: shape
                for key, shape in camera_infos.items()
                if key in xtrainer_policy_cameras
            }

        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos=camera_infos,
            task_type=task_type,
            policy_type=policy_type,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
        )
    elif "xtrainer_act" in args_cli.policy_type:
        from leisaac.policy import LeRobotServicePolicyClient
        from isaaclab.sensors import Camera

        model_type = 'xtrainer_act'

        camera_infos = {
            key: sensor.image_shape
            for key, sensor in env.scene.sensors.items()
            if isinstance(sensor, Camera)
        }
        if task_type == "xtrainerleader":
            xtrainer_policy_cameras = {"top", "left_wrist", "right_wrist"}
            camera_infos = {
                key: shape
                for key, shape in camera_infos.items()
                if key in xtrainer_policy_cameras
            }

        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos=camera_infos,
            task_type=task_type,
            policy_type='act',
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
        )
    elif args_cli.policy_type == "openpi":
        from leisaac.policy import OpenPIServicePolicyClient
        from isaaclab.sensors import Camera

        policy = OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            task_type=task_type,
        )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()

    # reset environment
    obs_dict, _ = env.reset()
    controller.reset()

    # record the results
    success_count, episode_count = 0, 1
    rollout_step = 0

    # simulate environment
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        success, time_out = False, False
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if controller.reset_state:
                    controller.reset()
                    obs_dict, _ = env.reset()
                    episode_count += 1
                    break

                obs_dict = preprocess_obs_dict(obs_dict['policy'], model_type, args_cli.policy_language_instruction)
                actions = policy.get_action(obs_dict).to(env.device)
                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]
                    action = maybe_apply_task3_first_grasp_protection(env, action, rollout_step)
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    append_task3_trajectory_log(env, rollout_step, action=action, source="policy")
                    rollout_step += 1
                    if reset_terminated[0]:
                        success = True
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)
            if success:
                print(f"[Evaluation] Episode {episode_count} is successful!")
                episode_count += 1
                success_count += 1
                break
            if time_out:
                print(f"[Evaluation] Episode {episode_count} timed out!")
                print_task3_timeout_diagnostics(env)
                episode_count += 1
                break
        print(f"[Evaluation] now success rate: {success_count / (episode_count - 1)}  [{success_count}/{episode_count - 1}]")
    print(f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f}  [{success_count}/{max_episode_count}]")

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()

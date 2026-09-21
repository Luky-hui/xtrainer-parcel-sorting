import pickle
import os
import json
import torch
import grpc
import time
import numpy as np

from .base import ZMQServicePolicy, Policy, WebsocketServicePolicy
from .lerobot.transport import services_pb2_grpc, services_pb2
from .lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from .lerobot.helpers import RemotePolicyConfig, TimedObservation
from .openpi import image_tools

from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES, XTRAINER_DUAL_JOINT_NAMES


class Gr00tServicePolicyClient(ZMQServicePolicy):
    """
    Service policy client for GR00T N1.5: https://github.com/NVIDIA/Isaac-GR00T
    Target Commit: https://github.com/NVIDIA/Isaac-GR00T/commit/b211007ed6698e6642d2fd7679dabab1d97e9e6c
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 5000,
        camera_keys: list[str] = ['front', 'wrist'],
        modality_keys: list[str] = ["single_arm", "gripper"],
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            timeout_ms: Timeout of the policy server.
            modality_keys: Keys of the modality.
        """
        super().__init__(host=host, port=port, timeout_ms=timeout_ms, ping_endpoint="ping")
        self.camera_keys = camera_keys
        self.modality_keys = modality_keys

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs_dict = {f"video.{key}": observation_dict[key].cpu().numpy().astype(np.uint8) for key in self.camera_keys}

        if "single_arm" in self.modality_keys:
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            obs_dict["state.single_arm"] = joint_pos[:, 0:5].astype(np.float64)
            obs_dict["state.gripper"] = joint_pos[:, 5:6].astype(np.float64)
        # TODO: add bi-arm support

        obs_dict["annotation.human.task_description"] = [observation_dict["task_description"]]

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "state.single_arm": np.zeros((1, 5)),
                "state.gripper": np.zeros((1, 1)),
                "annotation.human.action.task_description": [observation_dict["task_description"]],
            }
        """

        # get the action chunk via the policy server
        action_chunk = self.call_endpoint("get_action", obs_dict)

        """
            Example of action_chunk for single arm task:
            action_chunk = {
                "action.single_arm": np.zeros((1, 5)),
                "action.gripper": np.zeros((1, 1)),
            }
        """
        concat_action = np.concatenate(
            [action_chunk["action.single_arm"], action_chunk["action.gripper"][:, None]],
            axis=1,
        )
        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        return torch.from_numpy(concat_action[:, None, :])


class LeRobotServicePolicyClient(Policy):
    """
    Service policy client for Lerobot: https://github.com/huggingface/lerobot
    Target Commit: https://github.com/huggingface/lerobot/tree/v0.3.3
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 5000,
        camera_infos: dict[str, dict] = {},
        task_type: str = 'so101leader',
        policy_type: str = 'smolvla',
        pretrained_name_or_path: str = 'checkpoints/last/pretrained_model',
        actions_per_chunk: int = 50,
        device: str = 'cuda',
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            timeout_ms: Timeout of the policy server.
            camera_infos: List of camera information. {camera_key: (height, width)}
            task_type: Type of task.
            policy_type: Type of policy.
            pretrained_name_or_path: Path to the pretrained model in the remote policy server.
            actions_per_chunk: Number of actions per chunk.
            device: Device to use.
        """
        super().__init__("service")
        service_address = f'{host}:{port}'
        self.timeout_ms = timeout_ms
        self.task_type = task_type
        self.actions_per_chunk = actions_per_chunk
        self.action_stats_path = os.environ.get("XTRAINER_POLICY_ACTION_STATS_PATH")
        self.enable_gripper_latch = os.environ.get("XTRAINER_GRIPPER_LATCH", "0") == "1"
        self.gripper_close_threshold = float(os.environ.get("XTRAINER_GRIPPER_CLOSE_THRESHOLD", "0.012"))
        self.gripper_open_threshold = float(os.environ.get("XTRAINER_GRIPPER_OPEN_THRESHOLD", "0.006"))
        self.gripper_release_confirm = int(os.environ.get("XTRAINER_GRIPPER_RELEASE_CONFIRM", "10"))
        self.gripper_min_latch_steps = int(os.environ.get("XTRAINER_GRIPPER_MIN_LATCH_STEPS", "30"))
        self.gripper_latch_trigger_window = int(os.environ.get("XTRAINER_GRIPPER_LATCH_TRIGGER_WINDOW", str(actions_per_chunk)))
        self._gripper_latch_state = {
            "left": {"latched": False, "steps": 0, "open_count": 0},
            "right": {"latched": False, "steps": 0, "open_count": 0},
        }

        lerobot_features = {}
        self.last_action = None
        if task_type == 'so101leader':
            lerobot_features['observation.state'] = {
                'dtype': 'float32',
                'shape': (6,),
                'names': [f'{joint_name}.pos' for joint_name in SINGLE_ARM_JOINT_NAMES],
            }
            self.last_action = np.zeros((1, 6))
        # TODO: add bi-arm support
        if task_type == 'xtrainerleader':
            state_dim = 16
            lerobot_features['observation.state'] = {
                'dtype': 'float32',
                'shape': (state_dim,),
                'names': [f'{name}.pos' for name in XTRAINER_DUAL_JOINT_NAMES],
            }
            
            self.last_action = np.zeros((1, state_dim))
            print(f"[Client] Initialized XTrainer Dual Arm mode with {state_dim} joints.")

        for camera_key, camera_image_shape in camera_infos.items():
            lerobot_features[f'observation.images.{camera_key}'] = {
                'dtype': 'image',
                'shape': (camera_image_shape[0], camera_image_shape[1], 3),
                'names': ['height', 'width', 'channels'],
            }
        self.camera_keys = list(camera_infos.keys())

        self.policy_config = RemotePolicyConfig(
            policy_type,
            pretrained_name_or_path,
            lerobot_features,
            actions_per_chunk,
            device,
        )
        self.channel = grpc.insecure_channel(
            service_address, grpc_channel_options()
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.latest_action_step = 0

        self._init_service()

    @staticmethod
    def _xtrainer_gripper_values(action: np.ndarray) -> np.ndarray:
        return action[:, [6, 7, 14, 15]]

    def _append_action_stats(self, action: np.ndarray, stage: str) -> None:
        if not self.action_stats_path or self.task_type != "xtrainerleader":
            return
        gripper = self._xtrainer_gripper_values(action)
        record = {
            "timestep": self.latest_action_step,
            "stage": stage,
            "shape": list(action.shape),
            "gripper_names": ["J1_7", "J1_8", "J2_7", "J2_8"],
            "gripper_min": gripper.min(axis=0).tolist(),
            "gripper_max": gripper.max(axis=0).tolist(),
            "gripper_mean": gripper.mean(axis=0).tolist(),
            "gripper_q10": np.quantile(gripper, 0.10, axis=0).tolist(),
            "gripper_q50": np.quantile(gripper, 0.50, axis=0).tolist(),
            "gripper_q90": np.quantile(gripper, 0.90, axis=0).tolist(),
        }
        with open(self.action_stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _apply_xtrainer_gripper_latch(self, action: np.ndarray) -> np.ndarray:
        if not self.enable_gripper_latch or self.task_type != "xtrainerleader":
            return action

        action = action.copy()
        specs = {
            "left": (6, 7, -0.04, 0.04),
            "right": (14, 15, -0.04, 0.04),
        }
        for row_idx, row in enumerate(action):
            for side, (neg_idx, pos_idx, neg_closed, pos_closed) in specs.items():
                state = self._gripper_latch_state[side]
                close_signal = row[neg_idx] <= -self.gripper_close_threshold and row[pos_idx] >= self.gripper_close_threshold
                open_signal = abs(row[neg_idx]) <= self.gripper_open_threshold and abs(row[pos_idx]) <= self.gripper_open_threshold

                if not state["latched"] and close_signal and row_idx < self.gripper_latch_trigger_window:
                    state["latched"] = True
                    state["steps"] = 0
                    state["open_count"] = 0

                if state["latched"]:
                    row[neg_idx] = neg_closed
                    row[pos_idx] = pos_closed
                    state["steps"] += 1

                    if state["steps"] >= self.gripper_min_latch_steps and open_signal:
                        state["open_count"] += 1
                    else:
                        state["open_count"] = 0

                    if state["open_count"] >= self.gripper_release_confirm:
                        state["latched"] = False
                        state["steps"] = 0
                        state["open_count"] = 0
        return action

    def _init_service(self):
        try:
            self.stub.Ready(services_pb2.Empty())

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            print("Sending policy instructions to policy server, it may take a while...")
            self.stub.SendPolicyInstructions(policy_setup)
            print("Policy server is ready.")

        except grpc.RpcError:
            raise RuntimeError("Failed to connect to policy server")

    def _send_observation(self, observation_dict: dict):
        raw_observation = {f"{key}": observation_dict[key].cpu().numpy().astype(np.uint8)[0] for key in self.camera_keys}
        raw_observation["task"] = observation_dict["task_description"]

        if self.task_type == 'so101leader':
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            for joint_name in SINGLE_ARM_JOINT_NAMES:
                raw_observation[f"{joint_name}.pos"] = joint_pos[0, SINGLE_ARM_JOINT_NAMES.index(joint_name)].item()
        # TODO: add bi-arm support
        elif self.task_type == 'xtrainerleader':
            left_pos = observation_dict.get('left_joint_pos_rel')
            right_pos = observation_dict.get('right_joint_pos_rel')
            left_np = left_pos.cpu().numpy()   # (8,)
            right_np = right_pos.cpu().numpy() # (8,)
            joint_pos = np.concatenate([left_np, right_np], axis=1)

            # print(observation_dict.keys())

            for joint_name in XTRAINER_DUAL_JOINT_NAMES:
                raw_observation[f"{joint_name}.pos"] = joint_pos[0, XTRAINER_DUAL_JOINT_NAMES.index(joint_name)].item()

        """
            Example of raw_observation for single arm task:
            raw_observation = {
                "front": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 0.0,
                "task": "pick_and_place",
            }
        """
        self.latest_action_step += 1
        observation = TimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=self.latest_action_step,
            must_go=True,
        )

        # send observation to policy server
        observation_bytes = pickle.dumps(observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        _ = self.stub.SendObservations(observation_iterator)

    def _receive_action(self) -> dict:
        deadline = time.time() + self.timeout_ms / 1000.0
        while True:
            actions_chunk = self.stub.GetActions(services_pb2.Empty())
            if len(actions_chunk.data) != 0:
                return pickle.loads(actions_chunk.data)
            if time.time() >= deadline:
                print("Received `Empty` from policy server until timeout")
                return None
            time.sleep(0.02)

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        self._send_observation(observation_dict)
        action_chunk = self._receive_action()
        if action_chunk is None:
            print("[Info] Server returned Empty, holding position...")
            if self.last_action is None:
                return torch.zeros((self.actions_per_chunk, 1, 16), device=self.policy_config.device)
            return torch.from_numpy(self.last_action).repeat(self.actions_per_chunk, 1)[:, None, :]

        action_list = [action.get_action()[None, :] for action in action_chunk]
        concat_action = torch.cat(action_list, dim=0)

        if self.task_type == 'so101leader':
            concat_action = convert_lerobot_action_to_leisaac(concat_action)
        elif self.task_type == 'xtrainerleader':
            if isinstance(concat_action, torch.Tensor):
                concat_action = concat_action.cpu().numpy()

        if os.environ.get("XTRAINER_POLICY_VERBOSE_ACTIONS", "0") == "1":
            print(np.array2string(concat_action, precision=2, suppress_small=True, floatmode='fixed'))

        self._append_action_stats(concat_action, "raw")
        concat_action = self._apply_xtrainer_gripper_latch(concat_action)
        self._append_action_stats(concat_action, "post_latch")

        self.last_action = concat_action[-1, :]

        return torch.from_numpy(concat_action[:, None, :])


class OpenPIServicePolicyClient(WebsocketServicePolicy):
    """
    Service policy client for OpenPI: https://github.com/Physical-Intelligence/openpi
    Target Commit: https://github.com/Physical-Intelligence/openpi/commit/5bff19b0c0c447c7a7eaaaccf03f36d50998ec9d
    Reference: https://github.com/EverNorif/openpi/tree/lerobot-v0.3.3
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        camera_keys: list[str] = ['front', 'wrist'],
        task_type: str = "so101leader",
        api_key: str = None,
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            task_type: Type of task.
            api_key: API key of the policy server.
        """
        super().__init__(host=host, port=port, api_key=api_key)
        self.camera_keys = camera_keys
        self.task_type = task_type

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs_dict = {
            f"images/{key}": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(observation_dict[key].cpu().squeeze().numpy(), 224, 224)) for key in self.camera_keys}

        if self.task_type == 'so101leader':
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            obs_dict['state'] = joint_pos.squeeze().astype(np.float64)
        # TODO: add bi-arm support

        obs_dict["prompt"] = observation_dict["task_description"]

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "images/front": np.zeros((224, 224, 3), dtype=np.uint8),
                "images/wrist": np.zeros((224, 224, 3), dtype=np.uint8),
                "state": np.zeros(6),
                "prompt": observation_dict["task_description"],
            }
        """

        # get the action chunk via the policy server
        action_chunk = self.infer(obs_dict)['actions']

        """
            Example of action_chunk for single arm task:
            action_chunk: np.zeros((10, 6))
        """
        processed_action = convert_lerobot_action_to_leisaac(action_chunk)

        return torch.from_numpy(processed_action[:, None, :])

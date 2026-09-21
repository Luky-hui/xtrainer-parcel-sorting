import torch
from pathlib import Path

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from leisaac.utils.constant import ASSETS_ROOT
from leisaac.utils.domain_randomization import domain_randomization
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from . import mdp
from ..template import XTrainerArmTaskEnvCfg, XTrainerArmTaskSceneCfg, XTrainerArmTerminationsCfg, XTrainerArmObservationsCfg


TRAINING_ENV_USD_PATH = str(Path(ASSETS_ROOT) / "scenes" / "task3" / "training_env.usd")

TABLE_WITH_GOODS_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRAINING_ENV_USD_PATH,
    )
)

# ------------------------------------------------------------------------------
# Task3 可视化与成功判定参数
# - 4 个箱子都参与任务判定
# - ChinaBox / ChinaBox_01 需要放进 china_zone
# - AmericaBox / UKBox 需要放进 western_zone
# ------------------------------------------------------------------------------
TASK3_VISUALIZATION_CONTAINER_NAME = "scene"
TASK3_VISUALIZATION_OBJECT_NAMES = (
    "ChinaBox",
    "ChinaBox_01",
    "AmericaBox",
    "UKBox",
)

TASK3_VISUALIZATION_X_RANGE = (-0.10, 0.10)
TASK3_VISUALIZATION_Y_RANGE = (-0.10, 0.10)
TASK3_VISUALIZATION_HEIGHT_RANGE = (-0.05, 0.05)
TASK3_VISUALIZATION_OBJECT_OFFSETS = {
    # 说明：这三个箱子在 USD 里视觉中心相对根坐标有系统偏差，
    # 这里用局部坐标补偿到与 AmericaBox 一致的视觉中心参考。
    "ChinaBox": (4.3057, -3.9465, 0.02607),
    "ChinaBox_01": (4.30873, -3.94194, 0.0206),
    "AmericaBox": (0.0, 0.0, 0.0),
    "UKBox": (4.30654, -3.96032, 0.02865),
}

TASK3_VISUALIZATION_OBJECT_REGIONS = {
    "china_zone": {
        "x_range": (0.61, 0.79),
        "y_range": (-0.06, 0.2),
        "height_range": (-0.05, 0.18),
    },
    "western_zone": {
        "x_range": (0.26, 0.44),
        "y_range": (-0.06, 0.2),
        "height_range": (-0.05, 0.18),
    },
}

TASK3_VISUALIZATION_OBJECT_REGION_MAP = {
    "ChinaBox": "china_zone",
    "ChinaBox_01": "china_zone",
    "AmericaBox": "western_zone",
    "UKBox": "western_zone",
}

TASK3_OFFSET_FRAME = "local"  # "world" 或 "local"

TASK3_SUCCESS_CONTAINER_NAME = TASK3_VISUALIZATION_CONTAINER_NAME
TASK3_SUCCESS_OBJECT_NAMES = TASK3_VISUALIZATION_OBJECT_NAMES
TASK3_SUCCESS_X_RANGE = TASK3_VISUALIZATION_X_RANGE
TASK3_SUCCESS_Y_RANGE = TASK3_VISUALIZATION_Y_RANGE
TASK3_SUCCESS_HEIGHT_RANGE = TASK3_VISUALIZATION_HEIGHT_RANGE
TASK3_SUCCESS_OBJECT_OFFSETS = TASK3_VISUALIZATION_OBJECT_OFFSETS
TASK3_SUCCESS_OBJECT_REGIONS = TASK3_VISUALIZATION_OBJECT_REGIONS
TASK3_SUCCESS_OBJECT_REGION_MAP = TASK3_VISUALIZATION_OBJECT_REGION_MAP
TASK3_SUCCESS_OFFSET_FRAME = TASK3_OFFSET_FRAME


@configclass
class PickupRecognitionSceneCfg(XTrainerArmTaskSceneCfg):
    """Task3 场景定义。"""

    scene: AssetBaseCfg = TABLE_WITH_GOODS_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6",
                name="left_flange",
            ),
        ],
    )

    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6",
                name="right_grasp_center",
            ),
        ],
    )


@configclass
class Task3TerminationsCfg(XTrainerArmTerminationsCfg):
    """Task3 终止配置。"""

    success = DoneTerm(
        func=mdp.task_done,
        params={
            "object_names": TASK3_SUCCESS_OBJECT_NAMES,
            "container_name": TASK3_SUCCESS_CONTAINER_NAME,
            "x_range": TASK3_SUCCESS_X_RANGE,
            "y_range": TASK3_SUCCESS_Y_RANGE,
            "height_range": TASK3_SUCCESS_HEIGHT_RANGE,
            "object_offsets": TASK3_SUCCESS_OBJECT_OFFSETS,
            "object_regions": TASK3_SUCCESS_OBJECT_REGIONS,
            "object_region_map": TASK3_SUCCESS_OBJECT_REGION_MAP,
            "offset_frame": TASK3_SUCCESS_OFFSET_FRAME,
            "verbose": True,
            "visualize": True,
        },
    )


@configclass
class Task3EnvCfg(XTrainerArmTaskEnvCfg):
    """Task3 总环境配置。"""

    scene: PickupRecognitionSceneCfg = PickupRecognitionSceneCfg(env_spacing=8.0)
    observations: XTrainerArmObservationsCfg = XTrainerArmObservationsCfg()
    terminations: Task3TerminationsCfg = Task3TerminationsCfg()
    enable_visualization: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.1)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.sim.dt = 1.0 / 120.0
        self.sim.physx.enable_ccd = True

        # 注册需要纳入 reset 管理的目标物体
        reset_managed_parts = list(TASK3_VISUALIZATION_OBJECT_NAMES)
        parse_usd_and_create_subassets(
            TRAINING_ENV_USD_PATH,
            self,
            specific_name_list=reset_managed_parts,
            pose_reference="scene_root",
        )

        random_parts = list(TASK3_VISUALIZATION_OBJECT_NAMES)
        random_opts = []
        if random_parts:
            random_opts.append(
                EventTermCfg(
                    func=reset_tube_grid,
                    mode="reset",
                    params={
                        "asset_names": random_parts,
                        "x_range": (-0.30, 0.30),
                        "y_range": (-0.10, 0.15),
                        "grid_shape": (2, 2),
                        "min_center_distance": 0.05,
                        "max_sample_attempts": 24,
                    },
                )
            )

        domain_randomization(self, random_options=random_opts)


def reset_tube_grid(
    env,
    env_ids: torch.Tensor,
    asset_names: list[str],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid_shape: tuple[int, int] = (2, 2),
    min_center_distance: float = 0.05,
    max_sample_attempts: int = 20,
):
    """将多个物体随机分配到网格中，尽量避免重叠。"""
    if not asset_names:
        return

    num_envs = env.num_envs
    if env_ids is None:
        env_ids = torch.arange(num_envs, device=env.device)

    num_reset = len(env_ids)
    rows, cols = grid_shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid grid size: rows={rows}, cols={cols}")
    num_cells = rows * cols

    if len(asset_names) > num_cells:
        raise ValueError(f"Asset count ({len(asset_names)}) exceeds grid cells ({num_cells})!")

    x_min, x_max = x_range
    y_min, y_max = y_range

    cell_width = (x_max - x_min) / cols
    cell_height = (y_max - y_min) / rows

    x_indices = torch.arange(cols, device=env.device).repeat(rows)
    y_indices = torch.arange(rows, device=env.device).repeat_interleave(cols)

    grid_centers_x = x_min + (x_indices + 0.5) * cell_width
    grid_centers_y = y_min + (y_indices + 0.5) * cell_height
    grid_centers = torch.stack([grid_centers_x, grid_centers_y], dim=-1)

    rand_noise = torch.rand((num_reset, num_cells), device=env.device)
    cell_indices = torch.argsort(rand_noise, dim=-1)[:, : len(asset_names)]
    assigned_centers = grid_centers[cell_indices]

    max_offset_x = cell_width * 0.20
    max_offset_y = cell_height * 0.20
    num_assets = len(asset_names)
    sampled_offsets = assigned_centers.clone()
    offset_scale = torch.tensor([max_offset_x, max_offset_y], device=env.device)

    effective_min_distance = min(min_center_distance, min(cell_width, cell_height) * 0.9)

    for env_idx in range(num_reset):
        centers = assigned_centers[env_idx]
        if num_assets == 1:
            noise = 2 * torch.rand((1, 2), device=env.device) - 1
            sampled_offsets[env_idx] = centers + noise * offset_scale
            continue

        success = False
        for _ in range(max_sample_attempts):
            noise = 2 * torch.rand((num_assets, 2), device=env.device) - 1
            candidate = centers + noise * offset_scale
            pairwise = torch.cdist(candidate.unsqueeze(0), candidate.unsqueeze(0)).squeeze(0)
            pairwise.fill_diagonal_(float("inf"))
            if torch.all(pairwise >= effective_min_distance):
                sampled_offsets[env_idx] = candidate
                success = True
                break

        if not success:
            sampled_offsets[env_idx] = centers

    for i, asset_name in enumerate(asset_names):
        asset = env.scene[asset_name]
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, 0] += sampled_offsets[:, i, 0]
        root_state[:, 1] += sampled_offsets[:, i, 1]
        root_state[:, 7:] = 0.0
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)

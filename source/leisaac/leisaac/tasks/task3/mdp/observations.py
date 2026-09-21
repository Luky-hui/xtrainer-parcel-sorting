import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

try:
    from pxr import UsdGeom, Gf, Usd, UsdShade, Sdf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    UsdShade = None
    Sdf = None


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.01,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    object_entity: RigidObject = env.scene[object_cfg.name]

    object_pos = object_entity.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    offset_local = torch.tensor([0.0, 0.0, 0.16], device=env.device).repeat(env.num_envs, 1)
    grasp_center_pos = ee_pos_w + quat_apply(ee_quat_w, offset_local)
    pos_diff = torch.linalg.vector_norm(object_pos - grasp_center_pos, dim=1)

    joint_ids, _ = robot.find_joints("J2_8")
    is_gripper_closed = robot.data.joint_pos[:, joint_ids[0]] > grasp_threshold
    return torch.logical_and(pos_diff < diff_threshold, is_gripper_closed)


def object_in_container(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    z_range: tuple[float, float] = (0.0, 0.20),
) -> torch.Tensor:
    object_entity: RigidObject = env.scene[object_cfg.name]
    container: RigidObject = env.scene[container_cfg.name]

    rel_pos = object_entity.data.root_pos_w - container.data.root_pos_w
    in_x = torch.logical_and(rel_pos[:, 0] > x_range[0], rel_pos[:, 0] < x_range[1])
    in_y = torch.logical_and(rel_pos[:, 1] > y_range[0], rel_pos[:, 1] < y_range[1])
    in_z = torch.logical_and(rel_pos[:, 2] > z_range[0], rel_pos[:, 2] < z_range[1])
    return torch.logical_and(torch.logical_and(in_x, in_y), in_z)


def get_object_pose_w(
    object_entity: RigidObject,
    env_id: int | None = None,
    prefer_link_frame: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = object_entity.data
    has_link_pose = hasattr(data, "root_link_pos_w") and hasattr(data, "root_link_quat_w")

    if prefer_link_frame and has_link_pose:
        pos_all = data.root_link_pos_w
        quat_all = data.root_link_quat_w
    else:
        pos_all = data.root_pos_w
        quat_all = data.root_quat_w

    if env_id is None:
        return pos_all, quat_all
    return pos_all[env_id], quat_all[env_id]


def _get_object_prim_path(
    object_entity: RigidObject,
    env_id: int = 0,
) -> str | None:
    try:
        prim_paths = object_entity.root_physx_view.prim_paths
        if not prim_paths:
            return None
        if env_id < len(prim_paths):
            return prim_paths[env_id]
        return prim_paths[0].replace("env_0", f"env_{env_id}")
    except Exception:
        return None


def get_object_detection_pose_w(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_entity: RigidObject,
    env_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = object_entity.data
    has_link_pose = hasattr(data, "root_link_pos_w") and hasattr(data, "root_link_quat_w")
    if has_link_pose:
        return get_object_pose_w(object_entity, env_id=env_id, prefer_link_frame=True)
    return get_object_pose_w(object_entity, env_id=env_id, prefer_link_frame=False)


def _get_scene_entity_pos_w(
    env: ManagerBasedRLEnv | DirectRLEnv,
    entity_name: str,
) -> torch.Tensor:
    entity = env.scene[entity_name]
    if isinstance(entity, RigidObject):
        pos_w, _ = get_object_detection_pose_w(env, entity, env_id=None)
        return pos_w

    data = getattr(entity, "data", None)
    if data is not None:
        if hasattr(data, "root_link_pos_w"):
            return data.root_link_pos_w
        if hasattr(data, "root_pos_w"):
            return data.root_pos_w
        if hasattr(data, "root_state_w"):
            return data.root_state_w[:, :3]
    return env.scene.env_origins


def _get_env_base_path(env: ManagerBasedRLEnv | DirectRLEnv, entity_name: str) -> str:
    try:
        entity = env.scene[entity_name]
        prim_paths = entity.root_physx_view.prim_paths
        if prim_paths:
            parts = prim_paths[0].split("/")
            if len(parts) >= 4:
                return "/".join(parts[:4])
    except Exception:
        pass
    return "/World/envs/env_0"


def _get_object_world_scale(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_entity: RigidObject,
    env_id: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not USD_AVAILABLE:
        return torch.ones(3, dtype=dtype)

    if not hasattr(env, "_task3_object_world_scale_cache"):
        env._task3_object_world_scale_cache = {}

    try:
        prim_paths = object_entity.root_physx_view.prim_paths
        if not prim_paths:
            return torch.ones(3, dtype=dtype)
        if env_id < len(prim_paths):
            prim_path = prim_paths[env_id]
        else:
            prim_path = prim_paths[0].replace("env_0", f"env_{env_id}")
    except Exception:
        return torch.ones(3, dtype=dtype)

    cache_key = (prim_path, env_id)
    if cache_key in env._task3_object_world_scale_cache:
        scale = env._task3_object_world_scale_cache[cache_key]
        return torch.tensor(scale, dtype=dtype)

    try:
        stage = env.sim.stage
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return torch.ones(3, dtype=dtype)
        world_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        x_axis = world_matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        y_axis = world_matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
        z_axis = world_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
        scale = (
            float(x_axis.GetLength()),
            float(y_axis.GetLength()),
            float(z_axis.GetLength()),
        )
    except Exception:
        scale = (1.0, 1.0, 1.0)

    env._task3_object_world_scale_cache[cache_key] = scale
    return torch.tensor(scale, dtype=dtype)


def _get_object_local_visual_center(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_entity: RigidObject,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not USD_AVAILABLE:
        return torch.zeros(3, dtype=dtype)

    if not hasattr(env, "_task3_object_local_visual_center_cache"):
        env._task3_object_local_visual_center_cache = {}

    prim_path = _get_object_prim_path(object_entity, env_id=0)
    if prim_path is None:
        return torch.zeros(3, dtype=dtype)
    if prim_path in env._task3_object_local_visual_center_cache:
        cached = env._task3_object_local_visual_center_cache[prim_path]
        return torch.tensor(cached, dtype=dtype)

    center = (0.0, 0.0, 0.0)
    raw_center = (0.0, 0.0, 0.0)
    try:
        stage = env.sim.stage
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                useExtentsHint=True,
            )
            bbox = None
            try:
                bbox = bbox_cache.ComputeUntransformedBound(prim)
            except Exception:
                try:
                    bbox = bbox_cache.ComputeLocalBound(prim)
                except Exception:
                    bbox = None
            if bbox is not None:
                bbox_range = bbox.GetRange()
                if bbox_range is not None and not bbox_range.IsEmpty():
                    midpoint = bbox_range.GetMidpoint()
                    raw_center = (float(midpoint[0]), float(midpoint[1]), float(midpoint[2]))
                    center = raw_center
    except Exception:
        center = (0.0, 0.0, 0.0)

    # 部分 USD 资产会返回“世界量级”中点（而非局部），这里自动纠正回局部坐标
    try:
        candidate_a = torch.tensor(center, dtype=dtype)
        candidate_b = None

        if torch.linalg.vector_norm(candidate_a) > 0.25:
            stage = env.sim.stage
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                world_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                world_center = Gf.Vec3d(*raw_center)
                local_center_b = world_matrix.GetInverse().Transform(world_center)
                candidate_b = torch.tensor(
                    (float(local_center_b[0]), float(local_center_b[1]), float(local_center_b[2])),
                    dtype=dtype,
                )

        chosen = candidate_a
        if candidate_b is not None and torch.linalg.vector_norm(candidate_b) < torch.linalg.vector_norm(candidate_a):
            chosen = candidate_b

        center = (float(chosen[0]), float(chosen[1]), float(chosen[2]))
    except Exception:
        pass

    env._task3_object_local_visual_center_cache[prim_path] = center
    return torch.tensor(center, dtype=dtype)


def get_detection_point_pos_w(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_entity: RigidObject,
    object_offset: tuple[float, float, float],
    offset_frame: str = "local",
    apply_object_scale: bool = True,
    use_visual_center_anchor: bool = True,
) -> torch.Tensor:
    object_pos_w, object_quat_w = get_object_detection_pose_w(env, object_entity, env_id=None)
    point_pos_w = object_pos_w.clone()
    use_local_offset = offset_frame == "local"

    scale_all = None
    if apply_object_scale:
        scale_list = [_get_object_world_scale(env, object_entity, env_id, object_pos_w.dtype) for env_id in range(env.num_envs)]
        scale_all = torch.stack(scale_list, dim=0).to(device=env.device)

    if use_visual_center_anchor:
        local_center = _get_object_local_visual_center(env, object_entity, object_pos_w.dtype).to(device=env.device)
        if torch.linalg.vector_norm(local_center) > 0.0:
            anchor_local = local_center.unsqueeze(0).repeat(env.num_envs, 1)
            if apply_object_scale and scale_all is not None:
                anchor_local = anchor_local * scale_all
            anchor_world = quat_apply(object_quat_w, anchor_local)
            point_pos_w = point_pos_w + anchor_world

    offset_tensor = torch.tensor(object_offset, device=env.device, dtype=object_pos_w.dtype).unsqueeze(0)
    offset_tensor = offset_tensor.repeat(env.num_envs, 1)
    if use_local_offset:
        if apply_object_scale and scale_all is not None:
            offset_tensor = offset_tensor * scale_all
        offset_world = quat_apply(object_quat_w, offset_tensor)
    else:
        offset_world = offset_tensor

    return point_pos_w + offset_world


def _compute_detection_point_world_pos(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_entity: RigidObject,
    env_id: int,
    object_offset: tuple[float, float, float],
    offset_frame: str = "local",
    apply_object_scale: bool = True,
    use_visual_center_anchor: bool = True,
) -> tuple[float, float, float]:
    point_pos_w = get_detection_point_pos_w(
        env=env,
        object_entity=object_entity,
        object_offset=object_offset,
        offset_frame=offset_frame,
        apply_object_scale=apply_object_scale,
        use_visual_center_anchor=use_visual_center_anchor,
    )[env_id].cpu()
    return (float(point_pos_w[0]), float(point_pos_w[1]), float(point_pos_w[2]))


def _set_or_create_translate_op_from_world(prim, world_pos: tuple[float, float, float]) -> None:
    local_pos = Gf.Vec3d(*world_pos)
    try:
        parent_prim = prim.GetParent()
        if parent_prim and parent_prim.IsValid():
            parent_world = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_pos = parent_world.GetInverse().Transform(local_pos)
    except Exception:
        local_pos = Gf.Vec3d(*world_pos)

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(local_pos)


def _bind_preview_surface(stage, target_prim, material_path: str, color: tuple[float, float, float], opacity: float) -> None:
    if UsdShade is None or Sdf is None:
        return
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(target_prim).Bind(material)


def create_detection_zone_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    container_cfg: SceneEntityCfg,
    container_name: str,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    z_range: tuple[float, float] = (-0.05, 0.05),
    color: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> bool:
    if not USD_AVAILABLE:
        return False

    try:
        stage = env.sim.stage
        _ = env.scene[container_cfg.name]
    except Exception:
        return False

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    box_size = (x_max - x_min, y_max - y_min, z_max - z_min)
    box_center = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)

    env_base_path = _get_env_base_path(env, container_cfg.name)
    container_pos = _get_scene_entity_pos_w(env, container_cfg.name)

    for env_id in range(env.num_envs):
        pos = container_pos[env_id].cpu()
        zone_world_pos = (float(pos[0] + box_center[0]), float(pos[1] + box_center[1]), float(pos[2] + box_center[2]))

        env_path = env_base_path.replace("env_0", f"env_{env_id}")
        zone_path = f"{env_path}/Scene/DetectionZone_{container_name}"
        if stage.GetPrimAtPath(zone_path).IsValid():
            stage.RemovePrim(zone_path)

        zone_xform = UsdGeom.Xform.Define(stage, zone_path).GetPrim()
        _set_or_create_translate_op_from_world(zone_xform, zone_world_pos)

        cube = UsdGeom.Cube.Define(stage, f"{zone_path}/Cube")
        cube.GetSizeAttr().Set(1.0)
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

        cube_xformable = UsdGeom.Xformable(cube.GetPrim())
        cube_xformable.ClearXformOpOrder()
        cube_xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*box_size))

        try:
            _bind_preview_surface(stage, cube.GetPrim(), f"{zone_path}/Material", color, opacity=0.25)
        except Exception:
            pass

    return True


def create_object_detection_point_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg,
    object_name: str,
    object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    offset_frame: str = "local",
    apply_object_scale: bool = True,
    use_visual_center_anchor: bool = True,
    color: tuple[float, float, float] = (1.0, 0.84, 0.0),
    radius: float = 0.015,
) -> bool:
    if not USD_AVAILABLE:
        return False

    try:
        stage = env.sim.stage
        object_entity: RigidObject = env.scene[object_cfg.name]
    except Exception:
        return False

    env_base_path = _get_env_base_path(env, object_cfg.name)
    use_offset_frame = "local" if offset_frame == "local" else "world"
    prim_paths: list[str] = []

    for env_id in range(env.num_envs):
        point_world_pos = _compute_detection_point_world_pos(
            env,
            object_entity,
            env_id,
            object_offset,
            use_offset_frame,
            apply_object_scale=apply_object_scale,
            use_visual_center_anchor=use_visual_center_anchor,
        )
        env_path = env_base_path.replace("env_0", f"env_{env_id}")
        point_path = f"{env_path}/Scene/DetectionPoint_{object_name}"
        prim_paths.append(point_path)

        if stage.GetPrimAtPath(point_path).IsValid():
            stage.RemovePrim(point_path)

        point_xform = UsdGeom.Xform.Define(stage, point_path).GetPrim()
        _set_or_create_translate_op_from_world(point_xform, point_world_pos)

        sphere = UsdGeom.Sphere.Define(stage, f"{point_path}/Sphere")
        sphere.GetRadiusAttr().Set(radius)
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

        try:
            _bind_preview_surface(stage, sphere.GetPrim(), f"{point_path}/Material", color, opacity=1.0)
        except Exception:
            pass

    if not hasattr(env, "_task3_detection_point_visualizations"):
        env._task3_detection_point_visualizations = {}

    env._task3_detection_point_visualizations[object_name] = {
        "object_cfg": object_cfg,
        "object_offset": object_offset,
        "offset_frame": use_offset_frame,
        "apply_object_scale": apply_object_scale,
        "use_visual_center_anchor": use_visual_center_anchor,
        "prim_paths": prim_paths,
    }
    return True


def update_object_detection_point_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_name: str,
) -> None:
    if not USD_AVAILABLE or not hasattr(env, "_task3_detection_point_visualizations"):
        return
    if object_name not in env._task3_detection_point_visualizations:
        return

    vis_info = env._task3_detection_point_visualizations[object_name]
    object_entity: RigidObject = env.scene[vis_info["object_cfg"].name]
    object_offset = vis_info["object_offset"]
    offset_frame = vis_info.get("offset_frame", "local")
    apply_object_scale = vis_info.get("apply_object_scale", True)
    use_visual_center_anchor = vis_info.get("use_visual_center_anchor", True)
    stage = env.sim.stage

    for env_id, prim_path in enumerate(vis_info["prim_paths"]):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        point_world_pos = _compute_detection_point_world_pos(
            env,
            object_entity,
            env_id,
            object_offset,
            offset_frame,
            apply_object_scale=apply_object_scale,
            use_visual_center_anchor=use_visual_center_anchor,
        )
        _set_or_create_translate_op_from_world(prim, point_world_pos)


def init_task3_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    container_name: str,
    object_names: tuple[str, ...],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height_range: tuple[float, float],
    object_offsets: dict[str, tuple[float, float, float]] | None = None,
    object_regions: dict[str, dict[str, tuple[float, float]]] | None = None,
    object_region_map: dict[str, str] | None = None,
    offset_frame: str = "local",
    apply_object_scale: bool = True,
    use_visual_center_anchor: bool = True,
) -> None:
    if hasattr(env, "_task3_visualization_created"):
        return

    offsets = object_offsets or {}
    use_offset_frame = "local" if offset_frame == "local" else "world"

    if object_regions:
        zone_colors = (
            (0.0, 1.0, 0.0),
            (0.1, 0.7, 1.0),
            (1.0, 0.4, 0.0),
            (1.0, 0.8, 0.2),
        )
        for idx, (region_name, region_cfg) in enumerate(object_regions.items()):
            region_x_range = region_cfg.get("x_range", x_range)
            region_y_range = region_cfg.get("y_range", y_range)
            region_h_range = region_cfg.get("height_range", height_range)
            create_detection_zone_visualization(
                env=env,
                container_cfg=SceneEntityCfg(container_name),
                container_name=f"{container_name}_{region_name}",
                x_range=region_x_range,
                y_range=region_y_range,
                z_range=region_h_range,
                color=zone_colors[idx % len(zone_colors)],
            )
    else:
        create_detection_zone_visualization(
            env=env,
            container_cfg=SceneEntityCfg(container_name),
            container_name=container_name,
            x_range=x_range,
            y_range=y_range,
            z_range=height_range,
            color=(0.0, 1.0, 0.0),
        )

    region_point_colors = {
        "china_zone": (1.0, 0.84, 0.0),
        "western_zone": (0.1, 0.8, 1.0),
    }
    default_color = (1.0, 0.84, 0.0)
    for object_name in object_names:
        region_name = object_region_map.get(object_name) if object_region_map else None
        point_color = region_point_colors.get(region_name, default_color)
        create_object_detection_point_visualization(
            env=env,
            object_cfg=SceneEntityCfg(object_name),
            object_name=object_name,
            object_offset=offsets.get(object_name, (0.0, 0.0, 0.0)),
            offset_frame=use_offset_frame,
            apply_object_scale=apply_object_scale,
            use_visual_center_anchor=use_visual_center_anchor,
            color=point_color,
            radius=0.012,
        )

    env._task3_visualization_created = True


def update_task3_visualization(env: ManagerBasedRLEnv | DirectRLEnv) -> None:
    if not hasattr(env, "_task3_detection_point_visualizations"):
        return
    for object_name in tuple(env._task3_detection_point_visualizations.keys()):
        update_object_detection_point_visualization(env, object_name)

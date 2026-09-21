from pxr import Usd
from pxr import UsdPhysics
from pxr import UsdGeom


def get_all_prims(stage, prim=None, prims_list=None):
    if prims_list is None:
        prims_list = []
    if prim is None:
        prim = stage.GetPseudoRoot()
    for child in prim.GetChildren():
        prims_list.append(child)
        get_all_prims(stage, child, prims_list)
    return prims_list


def classify_prim(prim):
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return "Articulation"
    elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return "RigidBody"
    else:
        return "Normal"


def is_articulation_root(prim):
    return prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def is_rigidbody(prim):
    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


def get_all_joints(stage):
    joints = []

    def recurse(prim):
        if UsdPhysics.Joint(prim):
            joints.append(prim)
        for child in prim.GetChildren():
            recurse(child)
    recurse(stage.GetPseudoRoot())
    return joints


def get_stage(usd_path):
    stage = Usd.Stage.Open(usd_path)
    return stage


def _matrix_to_pos_rot(matrix):
    if matrix.Orthonormalize(issueWarning=True):
        rot = matrix.ExtractRotationQuat()
        rot_list = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]]
    else:
        rot_list = [1, 0, 0, 0]
    pos = matrix.ExtractTranslation()
    pos_list = list(pos)
    return pos_list, rot_list


def _get_top_level_prim(prim):
    path = prim.GetPath().pathString
    parts = [part for part in path.split("/") if part]
    if not parts:
        return None
    root_path = f"/{parts[0]}"
    return prim.GetStage().GetPrimAtPath(root_path)


def get_prim_pos_rot(prim, reference="world"):
    """
    获取 prim 位姿。

    说明：
    - world: prim 世界坐标；
    - parent: 相对父节点；
    - scene_root: 相对 USD 顶层根 prim（常用于被整体 reference 到环境时）。
    """
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return None, None
    time_code = Usd.TimeCode.Default()
    world_matrix = xformable.ComputeLocalToWorldTransform(time_code)

    if reference == "world":
        return _matrix_to_pos_rot(world_matrix)

    if reference == "parent":
        parent = prim.GetParent()
        if parent and parent.IsValid():
            parent_xformable = UsdGeom.Xformable(parent)
            if parent_xformable:
                parent_world_matrix = parent_xformable.ComputeLocalToWorldTransform(time_code)
                local_matrix = world_matrix * parent_world_matrix.GetInverse()
                return _matrix_to_pos_rot(local_matrix)
        return _matrix_to_pos_rot(world_matrix)

    if reference == "scene_root":
        root_prim = _get_top_level_prim(prim)
        if root_prim and root_prim.IsValid():
            root_xformable = UsdGeom.Xformable(root_prim)
            if root_xformable:
                root_world_matrix = root_xformable.ComputeLocalToWorldTransform(time_code)
                relative_matrix = world_matrix * root_world_matrix.GetInverse()
                return _matrix_to_pos_rot(relative_matrix)
        return _matrix_to_pos_rot(world_matrix)

    raise ValueError(f"Unsupported pose reference: {reference}")


def get_articulation_joints(articulation_prim):
    joints = []

    def recurse(prim):
        if UsdPhysics.Joint(prim):
            joints.append(prim)
        for child in prim.GetChildren():
            recurse(child)
    recurse(articulation_prim)
    return joints


def get_joint_type(joint_prim):
    joint = UsdPhysics.Joint(joint_prim)
    return joint.GetTypeName()


def is_fixed_joint(prim):
    return prim.GetTypeName() == 'PhysicsFixedJoint'


def is_revolute_joint(prim):
    return prim.GetTypeName() == 'PhysicsRevoluteJoint'


def is_prismatic_joint(prim):
    return prim.GetTypeName() == "PhysicsPrismaticJoint"


def get_joint_name_and_qpos(joint_prim):
    joint = UsdPhysics.Joint(joint_prim)
    return joint.GetName(), joint.GetPositionAttr().Get()


def get_all_joints_without_fixed(articulation_prim):
    joints = get_articulation_joints(articulation_prim)
    return [joint for joint in joints if not is_fixed_joint(joint)]


from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg

from isaaclab.sim.utils import clone

import isaacsim.core.utils.prims as prim_utils


def _match_name(prim_path, name_list):
    if name_list is None:
        return False
    prim_name = prim_path.split("/")[-1]
    for name in name_list:
        if not name:
            continue
        if name == prim_name or name == prim_path:
            return True
    return False


def match_specific_name(prim_path, specific_name_list, exclude_name_list):
    match_specific = True if specific_name_list is None else _match_name(prim_path, specific_name_list)
    match_exclude = False if exclude_name_list is None else _match_name(prim_path, exclude_name_list)

    return match_specific and not match_exclude


@clone
def spawn_from_prim_path(prim_path, spawn, translation, orientation):
    return prim_utils.get_prim_at_path(prim_path)


def parse_usd_and_create_subassets(
    usd_path,
    env_cfg,
    specific_name_list=None,
    exclude_name_list=None,
    pose_reference="world",
):
    stage = get_stage(usd_path)
    prims = get_all_prims(stage)
    articulation_sub_prims = list()
    create_attr_record = dict()
    for prim in prims:
        if prim.IsPrototype() or prim.IsInPrototype():
            continue
        if is_articulation_root(prim) and match_specific_name(prim.GetPath().pathString, specific_name_list, exclude_name_list):
            pos, rot = get_prim_pos_rot(prim, reference=pose_reference)
            joints = get_all_joints_without_fixed(prim)
            if not joints:
                continue
            orin_prim_path = prim.GetPath().pathString
            name = orin_prim_path.split("/")[-1]
            if name not in create_attr_record:
                create_attr_record[name] = 0
            else:
                create_attr_record[name] += 1
                name = f"{name}_{create_attr_record[name]}"
            sub_prim_path = orin_prim_path[orin_prim_path.find('/', 1) + 1:]
            prim_path = f"{{ENV_REGEX_NS}}/Scene/{sub_prim_path}"
            artcfg = ArticulationCfg(
                prim_path=prim_path,
                spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=pos,
                    rot=rot,
                ),
                actuators={},
            )
            setattr(env_cfg.scene, name, artcfg)
            articulation_sub_prims.extend(get_all_prims(stage, prim))
    for prim in prims:
        if prim.IsPrototype() or prim.IsInPrototype():
            continue
        if is_rigidbody(prim) and match_specific_name(prim.GetPath().pathString, specific_name_list, exclude_name_list):
            if prim in articulation_sub_prims:
                continue
            pos, rot = get_prim_pos_rot(prim, reference=pose_reference)
            orin_prim_path = prim.GetPath().pathString
            name = orin_prim_path.split("/")[-1]
            if name not in create_attr_record:
                create_attr_record[name] = 0
            else:
                create_attr_record[name] += 1
                name = f"{name}_{create_attr_record[name]}"
            sub_prim_path = orin_prim_path[orin_prim_path.find('/', 1) + 1:]
            prim_path = f"{{ENV_REGEX_NS}}/Scene/{sub_prim_path}"
            rigidcfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=RigidObjectSpawnerCfg(func=spawn_from_prim_path),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos,
                    rot=rot,
                ),
            )
            setattr(env_cfg.scene, name, rigidcfg)

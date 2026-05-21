"""UR5e + Robotiq 2F-85 articulated gripper scene config for tiger pick.

Uses ur5e_robotiq_articulated.usd (single articulation with working gripper).
No more _spawn_grippers hack needed — the gripper is part of the robot USD.
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiTiledCameraCfg
from isaaclab.utils import configclass

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

from real2render2real.isaaclab_viser.configs.articulation_configs.ur5e_robotiq_cfg import (
    UR5E_ROBOTIQ_CFG,
)

PILLAR_X = 0.0
PILLAR_Y = 0.0
PILLAR_Z = 0.0088

ROBOT1_POS = (-0.7458, 0.0652, 0.7786)
ROBOT1_ROT = (0.30827, 0.30863, 0.83179, -0.34329)

SIMPLE_ROOM_POS = (-0.2758352213446611, 0.2313240569149917, 0.07010507829325907)

D435I_INTRINSICS_720 = [910.51, 0, 644.55, 0, 910.20, 369.72, 0, 0, 1]


@configclass
class UR5eRobotiqTigerPickCfg(InteractiveSceneCfg):
    """Tiger pick with articulated Robotiq gripper."""

    robot: ArticulationCfg = UR5E_ROBOTIQ_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot1",
        init_state=UR5E_ROBOTIQ_CFG.init_state.replace(
            pos=ROBOT1_POS, rot=ROBOT1_ROT
        ),
    )

    pillar = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pillar",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/pillar/立柱装配体V2.5.usd",
            scale=(0.001, 0.001, 0.001),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(PILLAR_X, PILLAR_Y, PILLAR_Z)),
    )

    simple_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SimpleRoom",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Room/simple_room.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SIMPLE_ROOM_POS),
    )

    viewport_camera = MultiTiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Viewport",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=D435I_INTRINSICS_720,
            height=720,
            width=1280,
            clipping_range=(0.01, 20),
        ),
        cams_per_env=2,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=200.0),
    )

    dome_light2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light2",
        spawn=sim_utils.CylinderLightCfg(intensity=200.0, radius=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.46, -0.64, 1.0)),
    )

    dome_light3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light3",
        spawn=sim_utils.CylinderLightCfg(intensity=200.0, radius=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.46, 0.4, 1.0)),
    )

    tiger = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tiger",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/tiger/tiger_new.usd",
            scale=(1.1, 1.1, 1.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.40, 0.05, 0.12),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

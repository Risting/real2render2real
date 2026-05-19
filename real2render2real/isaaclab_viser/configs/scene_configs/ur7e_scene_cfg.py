"""UR5e chili pick scene configuration.

Lab setup:
- Two UR5e mounted on a structural pillar; only Robot1 is articulated
- Pillar is mm-scale CAD, converted to meters via scale=(0.001, 0.001, 0.001)
- Pillar internal CAD orientation corrected via post-spawn matrix (see base.py)
- Robotiq 2F-85 gripper attached as a static mesh under wrist_3_link
- SimpleRoom provides floor, walls, and built-in table

Pillar Z derived from bbox measurements (see ur7e_scene_cfg_test.py for details):
  pillar bbox min Z (scale 0.001) = 0.071727 m
  SimpleRoom table top world Z = 0.0805 m
  → PILLAR_Z = 0.0805 - 0.071727 = 0.0088 m

Robot world positions:
  world_pos = (PILLAR_X, PILLAR_Y, PILLAR_Z) + usda_translate * 0.001
  UR5E_R: (-0.7458, 0.0652, 0.7786)
  UR5E_L: (-0.7208, 0.4421, 0.7768)
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiTiledCameraCfg
from isaaclab.utils import configclass

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

from real2render2real.isaaclab_viser.configs.articulation_configs.ur7e_cfg import (
    UR5E_CFG,
)

# --- Pillar ---
PILLAR_X = 0.0
PILLAR_Y = 0.0
PILLAR_Z = 0.0088   # pillar base sits on SimpleRoom table surface

# --- Robot world positions (pillar Z + mm translate * 0.001) ---
ROBOT1_POS = (-0.7458, 0.0652, 0.7786)
ROBOT1_ROT = (0.30827, 0.30863, 0.83179, -0.34329)

ROBOT2_POS = (-0.7208, 0.4421, 0.7768)
ROBOT2_ROT = (-0.21841, 0.74088, 0.48853, 0.40586)

# --- Gripper wrist-local mount (from PhysicsFixedJoint + base_link offset) ---
GRIPPER_POS = (-0.00548, -0.00440, -0.02588)
GRIPPER_ROT = (0.70710677, 0.0, 0.0, -0.70710677)

# --- SimpleRoom position (verbatim from ur7e_test.usda) ---
SIMPLE_ROOM_POS = (-0.2758352213446611, 0.2313240569149917, 0.07010507829325907)

# --- Camera intrinsics ---
D435I_INTRINSICS_720 = [910.51, 0, 644.55, 0, 910.20, 369.72, 0, 0, 1]
D405_INTRINSICS = [653.62, 0, 634.20, 0, 652.66, 343.31, 0, 0, 1]


@configclass
class UR7eBaseCfg(InteractiveSceneCfg):
    """Base scene: Robot1 (active) + Robot2 (static visual) + pillar + SimpleRoom + lights."""

    robot: ArticulationCfg = UR5E_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot1",
        init_state=UR5E_CFG.init_state.replace(pos=ROBOT1_POS, rot=ROBOT1_ROT),
    )

    # Structural pillar; mm-scale CAD → meters via scale 0.001.
    # Internal CAD orientation fixed by _apply_pillar_orientation in ChiliPick.
    pillar = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pillar",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/pillar/立柱装配体V2.5.usd",
            scale=(0.001, 0.001, 0.001),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(PILLAR_X, PILLAR_Y, PILLAR_Z),
        ),
    )

    # Second robot (UR5E_L, left arm — static visual only, no articulation)
    robot2_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot2_Visual",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/ur7e_description/ur5e/ur5e.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=ROBOT2_POS, rot=ROBOT2_ROT),
    )

    # SimpleRoom (floor, walls, table)
    simple_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SimpleRoom",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Room/simple_room.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SIMPLE_ROOM_POS),
    )

    # Cameras (2 per env: cam_0 fixed + cam_1 wrist)
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

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=200.0,
        ),
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


@configclass
class UR7eChiliPickCfg(UR7eBaseCfg):
    """Chili pick task: UR5e picks up a chili from the table."""

    chili = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/chili",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/chili/chili.usdz",
            scale=(0.1, 0.1, 0.1),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.84),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

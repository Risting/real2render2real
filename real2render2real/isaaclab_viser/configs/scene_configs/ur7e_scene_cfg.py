"""UR5e chili pick scene configuration.

Lab setup:
- Two UR5e (proxy for UR7e) mounted upside-down on a structural pillar
- Table: 120cm x 80cm, surface height 79cm
- Pillar: 8cm from long edge, centered
- Only Robot1 (left arm) is active, Robot2 is a static visual prop
- Fixed camera: RealSense D435I
- Wrist camera: RealSense D405

Camera intrinsics from real cameras:
  Fixed camera (D435I): 1920x1080, fx=1365.77, fy=1365.30
  Wrist camera (D405): 1280x720, fx=653.62, fy=652.66
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

# --- Table (same table2 as yumi/franka setups) ---
TABLE_HEIGHT = 0.79  # 79cm to surface

# --- Pillar: origin at (0,0,0), matches /World/BODY in UR7E_2.usd ---
# CAD is in mm → scale 0.001 converts to m
PILLAR_POS = (0.0, 0.0, 0.0)

# --- Robot mounting ---
# UR5E_L from UR7E_2.usd (world coords after BODY's unitsResolve=0.001):
#   translate=(-720.82, 442.08, 768.02) mm → (-0.721, 0.442, 0.768) m
#   orient=(-0.2184, 0.7409, 0.4885, 0.4059)
ROBOT1_POS = (-0.721, 0.442, 0.768)
ROBOT1_ROT = (-0.2184, 0.7409, 0.4885, 0.4059)

# UR5E_R from UR7E_2.usd (static prop):
ROBOT2_POS = (-0.746, 0.065, 0.770)
ROBOT2_ROT = (0.3083, 0.3086, 0.8318, -0.3433)

# --- Camera intrinsics ---
# D435I: 1920x1080 -> scaled to 1280x720
# fx_scaled = 1365.77 * (1280/1920) = 910.51
# fy_scaled = 1365.30 * (720/1080) = 910.20
# ppx_scaled = 966.83 * (1280/1920) = 644.55
# ppy_scaled = 554.58 * (720/1080) = 369.72
D435I_INTRINSICS_720 = [910.51, 0, 644.55, 0, 910.20, 369.72, 0, 0, 1]

# D405: 1280x720 native
D405_INTRINSICS = [653.62, 0, 634.20, 0, 652.66, 343.31, 0, 0, 1]


@configclass
class UR7eBaseCfg(InteractiveSceneCfg):
    """Base scene: Robot1 (active) + Robot2 (static prop) + table + pillar + cameras."""

    # Active robot (articulation with joints, pos/rot in UR5E_CFG)
    robot: ArticulationCfg = UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")

    # Pillar
    pillar = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pillar",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/pillar/立柱装配体V2.5.usd",
            scale=(0.001, 0.001, 0.001),   # mm→m, matches BODY's unitsResolve
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=PILLAR_POS,
        ),
    )

    # Second robot (UR5E_L, left arm — static visual prop)
    robot2: ArticulationCfg = UR5E_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot2",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT2_POS,
            rot=ROBOT2_ROT,
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
            },
        ),
    )

    # SimpleRoom (floor, walls, table — from collected scene, replaces table2 + ground)
    # Position matches /SimpleRoom in UR7E_2.usd
    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Room/simple_room.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.276, 0.231, 0.070),
        ),
    )

    # Cameras (2 per env: fixed D435I + wrist D405)
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
        spawn=sim_utils.CylinderLightCfg(
            intensity=200.0,
            radius=1.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.46, -0.64, 1.0)),
    )

    dome_light3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light3",
        spawn=sim_utils.CylinderLightCfg(
            intensity=200.0,
            radius=1.0,
        ),
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
            pos=(0.4, 0.0, 0.05),          # just above table surface
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

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

# --- Table dimensions ---
TABLE_LENGTH = 1.2   # 120cm
TABLE_WIDTH = 0.8    # 80cm
TABLE_HEIGHT = 0.79  # 79cm to surface

# --- Pillar position ---
# Centered on the long axis (x=0), 8cm from one long edge
PILLAR_X = 0.0
PILLAR_Y = TABLE_WIDTH / 2 - 0.08  # 0.32m from center

# --- Robot mounting on pillar ---
# Extracted from UR7E_2.usd: rotation is IDENTITY (not upside-down).
# Robot base on pillar, at table height + pillar offset.
# CAD coords in reference are ~cm scale; we use meters scaled to our scene.
ROBOT1_POS = (PILLAR_X - 0.15, PILLAR_Y, TABLE_HEIGHT + 0.5)
ROBOT1_ROT = (1.0, 0.0, 0.0, 0.0)  # Identity — from UR7E_2.usd reference

ROBOT2_POS = (PILLAR_X + 0.15, PILLAR_Y, TABLE_HEIGHT + 0.5)
ROBOT2_ROT = (1.0, 0.0, 0.0, 0.0)

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
            scale=(0.01, 0.01, 0.01),   # CAD model likely in cm — convert to m
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(PILLAR_X, PILLAR_Y, 0.0),
        ),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/table2/table2_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                retain_accelerations=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, 0.0)),
    )

    # Gripper (DH AG-95, attached to wrist_3_link at tool0 position)
    # URDF tool0: offset (0, 0.1, 0) from wrist_3_link, rpy=(-90°, 0, 0)
    gripper = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot1/wrist_3_link/gripper",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/gripper/dh_ag_95_base.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.1, 0.0),
            rot=(-0.7071, 0.7071, 0.0, 0.0),
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

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.75)),
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
            pos=(0.4, 0.0, TABLE_HEIGHT + 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

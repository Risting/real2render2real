"""Spatial perception scene: SimpleRoom + pillar + red cube + 2 fixed cameras.

No robot — just a static scene with a randomly-placed cube and two cameras
capturing stereo-like views for spatial perception training data.
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiTiledCameraCfg
from isaaclab.utils import configclass

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

# --- Pillar ---
PILLAR_X = 0.0
PILLAR_Y = 0.0
PILLAR_Z = 0.0088   # pillar base sits on SimpleRoom table surface

# --- SimpleRoom position (verbatim from ur7e_test.usda) ---
SIMPLE_ROOM_POS = (-0.2758352213446611, 0.2313240569149917, 0.07010507829325907)

# --- Camera intrinsics (D435I at 720p) ---
D435I_INTRINSICS_720 = [910.51, 0, 644.55, 0, 910.20, 369.72, 0, 0, 1]

# --- Cube initial position (centre of randomisation volume) ---
CUBE_INIT_POS = (-0.40, 0.15, 0.18)


@configclass
class SpatialPerceptionCfg(InteractiveSceneCfg):
    """Scene with SimpleRoom, pillar backdrop, a red cube, and 2 fixed cameras."""

    # Structural pillar — mm-scale CAD → meters via scale 0.001
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

    # SimpleRoom (floor, walls, table)
    simple_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SimpleRoom",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Room/simple_room.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SIMPLE_ROOM_POS),
    )

    # Red cube — pure visual prim, no physics (moved via USD xform API)
    cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.15, 0.15),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=CUBE_INIT_POS),
    )

    # Two fixed cameras (cam_0 front-left, cam_1 front-right)
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

"""UR5e (proxy for UR7e) scene configuration.

Lab setup:
- Two UR5e mounted on a structural pillar; only Robot1 is articulated
- Pillar is mm-scale CAD, converted to meters via scale=(0.001, 0.001, 0.001)
- Robotiq 2F-85 gripper attached as a static mesh under wrist_3_link
- SimpleRoom provides the room environment (includes a built-in table)
- No separate table USD required

Asset paths resolve relative to this file's location inside ur7e_description,
so the package is portable to any directory.

NOTE — pillar internal CAD orientation:
  pillar_assembly.usd ships with an identity transform on its inner CAD node
  (tn__V25_V5xgg2sec0sYY0isSaiJ). The original ur7e_test.usda overrides it
  with a matrix4d that rotates the geometry upright. Isaac Lab's UsdFileCfg
  cannot inject sub-prim overrides at spawn time, so the pillar may appear
  tilted in simulation. To fix this permanently, bake the orientation into
  pillar_assembly.usd itself using export_cfg_to_usd.py, or apply the
  override via a post-spawn stage callback:
      mat = Gf.Matrix4d(
           0.5054585783556504, -0.01564235936930266, -0.862709071564713,  0,
           0.8614450409941848, -0.0479028782148914,   0.5055865461078429, 0,
          -0.04923481403654982, -0.9987295083515333, -0.010737887813508283, 0,
          -576.3422577523722,  710.4825268723347,    523.6951992568527,   1,
      )
      inner = stage.GetPrimAtPath("<pillar_prim>/tn__V25_V5xgg2sec0sYY0isSaiJ")
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

_dir_path = os.path.dirname(os.path.realpath(__file__))
_pkg_dir = os.path.realpath(os.path.join(_dir_path, "..", ".."))

UR5E_USD    = os.path.join(_pkg_dir, "meshes", "usd", "ur5e", "ur5e.usd")
PILLAR_USD  = os.path.join(_pkg_dir, "meshes", "usd", "pillar", "pillar_assembly.usd")
GRIPPER_USD = os.path.join(_pkg_dir, "meshes", "usd", "robotiq_2f85", "Robotiq_2F_85_edit.usd")
ROOM_USD    = os.path.join(_pkg_dir, "meshes", "usd", "simple_room", "simple_room.usd")

from ur7e_description.robots.ur5e.ur5e_cfg import UR5E_CFG

# --- Table dimensions (SimpleRoom built-in table) ---
TABLE_HEIGHT = 0.79  # 79cm to surface

# --- Pillar ---
# pillar_assembly.usd is mm-scale CAD; scale=(0.001,0.001,0.001) converts to meters.
# Measured via UsdGeom.BBoxCache (scale=0.001 applied):
#   pillar bbox min Z = 0.071727 m  (CAD origin is not at the physical bottom)
#   SimpleRoom /Root/table_low_327 bbox max Z = 0.0104 m (room-local)
#   SimpleRoom world Z offset = 0.07010507829325907 m
#   table top world Z = 0.07010507829325907 + 0.0104 = 0.08050508 m
#   PILLAR_Z = table_top_world_Z - pillar_bottom_Z = 0.08050508 - 0.071727 = 0.008778 m
PILLAR_X = 0.0
PILLAR_Y = 0.0
PILLAR_Z = 0.0088

# --- Robot world positions ---
# Derived from ur7e_test.usda: robots are children of BODY (the pillar).
# world_pos = (PILLAR_X, PILLAR_Y, PILLAR_Z) + usda_translate * 0.001
#   UR5E_R usda_translate = (-745.8, 65.2, 769.8) mm -> (-0.7458, 0.0652, 0.7698) m
#   UR5E_L usda_translate = (-720.8, 442.1, 768.0) mm -> (-0.7208, 0.4421, 0.7680) m
# Z shifted by delta = 0.0088 - (-0.1548) = +0.1636 m from original USDA values
ROBOT1_POS = (-0.7458, 0.0652, 0.7786)
ROBOT1_ROT = (0.30827, 0.30863, 0.83179, -0.34329)   # from ur7e_test.usda xformOp:orient

ROBOT2_POS = (-0.7208, 0.4421, 0.7768)
ROBOT2_ROT = (-0.21841, 0.74088, 0.48853, 0.40586)

# --- Gripper wrist-local mount ---
# Derived from PhysicsFixedJoint physics:localRot0 in ur7e_test.usda (rot Z by -90 deg)
# plus the base_link micro-offset observed under each AG_* prim.
GRIPPER_POS = (-0.00548, -0.00440, -0.02588)
GRIPPER_ROT = (0.70710677, 0.0, 0.0, -0.70710677)

# --- SimpleRoom position ---
# Extracted verbatim from ur7e_test.usda /SimpleRoom xformOp:translate
SIMPLE_ROOM_POS = (-0.2758352213446611, 0.2313240569149917, 0.07010507829325907)
SIMPLE_ROOM_ROT = (1.0, 0.0, 0.0, 0.0)


@configclass
class UR7eBaseCfg(InteractiveSceneCfg):
    """Base scene: Robot1 (active) + Robot2 (static prop) + pillar + SimpleRoom + lights."""

    robot: ArticulationCfg = UR5E_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot1",
        init_state=UR5E_CFG.init_state.replace(pos=ROBOT1_POS, rot=ROBOT1_ROT),
    )

    # Structural pillar; mm-scale CAD converted to meters via scale.
    # See module docstring for the internal CAD orientation caveat.
    pillar = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pillar",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PILLAR_USD,
            scale=(0.001, 0.001, 0.001),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(PILLAR_X, PILLAR_Y, PILLAR_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    robot2_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot2_Visual",
        spawn=sim_utils.UsdFileCfg(usd_path=UR5E_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=ROBOT2_POS, rot=ROBOT2_ROT),
    )

    gripper = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot1/wrist_3_link/gripper",
        spawn=sim_utils.UsdFileCfg(usd_path=GRIPPER_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=GRIPPER_POS, rot=GRIPPER_ROT),
    )

    gripper2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot2_Visual/wrist_3_link/gripper",
        spawn=sim_utils.UsdFileCfg(usd_path=GRIPPER_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=GRIPPER_POS, rot=GRIPPER_ROT),
    )

    simple_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SimpleRoom",
        spawn=sim_utils.UsdFileCfg(usd_path=ROOM_USD),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=SIMPLE_ROOM_POS,
            rot=SIMPLE_ROOM_ROT,
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.75)),
    )

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
    """Chili pick task variant: add a chili object on the table surface."""

    chili = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/chili",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",   # fill in path to chili.usdz
            scale=(0.1, 0.1, 0.1),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.4, 0.0, TABLE_HEIGHT + 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

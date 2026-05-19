"""Configuration for the UR5e arm (proxy for UR7e).

USD path resolves relative to this file's location inside ur7e_description:
    <this_file>/../../meshes/usd/ur5e/ur5e.usd

Joint names have _joint suffix. Body names have _link suffix. EE body: wrist_3_link.
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_dir_path = os.path.dirname(os.path.realpath(__file__))
_pkg_dir = os.path.realpath(os.path.join(_dir_path, "..", ".."))
_usd_path = os.path.join(_pkg_dir, "meshes", "usd", "ur5e", "ur5e.usd")

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # World position derived from ur7e_test.usda:
        #   pillar BODY at Z=0.0088m (bbox-measured: pillar bottom=0.0717m, table top=0.0805m world Z)
        #   UR5e_R usda_translate * 0.001 = (-0.7458, 0.0652, 0.7698)
        #   world pos = pillar_pos + robot_offset = (-0.7458, 0.0652, 0.7786)
        pos=(-0.7458, 0.0652, 0.7786),
        # Rotation extracted verbatim from ur7e_test.usda xformOp:orient on UR5E_R
        rot=(0.30827, 0.30863, 0.83179, -0.34329),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

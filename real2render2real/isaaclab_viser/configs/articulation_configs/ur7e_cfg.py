"""Configuration for the UR5e arm (proxy for UR7e).

Uses Isaac 5.1 native UR5e USD articulation.
Joint names: shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
wrist_1_joint, wrist_2_joint, wrist_3_joint.
EE body: wrist_3_link.
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_dir_path = os.path.dirname(os.path.realpath(__file__))
_data_dir = os.path.join(_dir_path, "../../../../data")

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_data_dir, "ur7e_description/ur5e/ur5e.usd"),
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
        pos=(-0.746, 0.065, 0.770),        # UR5E_R from UR7E_2.usd (BODY mm→m)
        rot=(0.3083, 0.3086, 0.8318, -0.3433),  # UR5E_R orient from UR7E_2.usd
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

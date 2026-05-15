"""Configuration for the UR5e arm (proxy for UR7e).

Uses the pre-converted USD from the official UR5e URDF.
For the IK controller, uses the URDF at data/ur5e_robotiq/ur5e.urdf.
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_dir_path = os.path.dirname(os.path.realpath(__file__))
_data_dir = os.path.join(_dir_path, "../../../../data")

##
# Configuration
##

# UR5e arm only (no gripper articulation).
# USD path points to the pre-converted UR5e URDF on the server.
# The gripper will be added as a visual attachment in the scene config.
UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_data_dir, "ur7e_description/ur5e_urdf/ur5e.urdf.usd"),
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
"""UR5e arm (proxy for UR7e) without gripper articulation."""

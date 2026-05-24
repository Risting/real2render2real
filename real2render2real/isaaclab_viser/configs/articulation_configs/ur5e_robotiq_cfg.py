"""Configuration for UR5e + Robotiq 2F-85 as a single flat-hierarchy articulation.

Uses the flat USD from export_ur5e_robotiq_flat.py.
Joints: 6 arm + finger_joint (1-DOF gripper, mimic joints follow automatically).
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_dir_path = os.path.dirname(os.path.realpath(__file__))
_data_dir = os.path.join(_dir_path, "../../../../data")

UR5E_ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_data_dir, "ur7e_description/scenes/ur5e_robotiq_flat.usd"),
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
        pos=(-0.7458, 0.0652, 0.7786),
        rot=(0.30827, 0.30863, 0.83179, -0.34329),
        joint_pos={
            "shoulder_pan_joint": 0.9694,
            "shoulder_lift_joint": -2.4357,
            "elbow_joint": 2.2427,
            "wrist_1_joint": -0.5244,
            "wrist_2_joint": -4.1204,
            "wrist_3_joint": 1.6110,
            "finger_joint": 0.0,  # 0 = open, ~0.8 = closed
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
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            velocity_limit=2.0,
            effort_limit=16.5,
            stiffness=200.0,
            damping=10.0,
        ),
    },
)

"""Camera extrinsics for UR7e simulation cameras.

Source: sim_data/eye_hand/calibration_result_61.json (2026-06-01, tsai, 50/50 valid samples)

Coordinate frame chain:
  T_world_cam = T_world_blInertia @ T_blInertia_cam
  where T_world_blInertia = T_world_baseLink @ T_baseLink_baseLinkInertia(Rz(pi))
  Real robot base = base_link_inertia = teach pendant [ji zuo] mode

Calibration convention: T_A_B means p_A = T_A_B @ p_B
Camera frame is ROS/OpenCV (+Z forward, +X right, +Y down)
"""

import torch
from isaaclab.utils.math import quat_from_matrix

# --- Fixed camera (D435I) in world frame ---
# Computed from calibration T_base_cam_fixed in base_link_inertia frame,
# converted to world via R_world_blInertia = R_world_baseLink @ Rz(pi).
# base_pos=(-0.7458,0.0652,0.7786), base_quat=(0.30827,0.30863,0.83179,-0.34329)[wxyz]
FIXED_CAM_POS = [0.110814, 0.198867, 0.267785]
# Full rotation from calibration, converted to world. ROS/OpenCV convention.
FIXED_CAM_QUAT_WXYZ = [-0.371358, 0.515038, 0.573580, -0.517527]

# T_ee_cam_end: wrist camera (D405) in end-effector frame (from calibration 61)
WRIST_CAM_POS = [-0.007263472445299419, -0.06243658543806627, -0.1507997443408004]
WRIST_CAM_ROT = [
    [0.9991263436947527, 0.03907170953576453, -0.014830740005405345],
    [-0.038427258140137734, 0.998399255332287, 0.04150027450229563],
    [0.016428486448431645, -0.04089411285121023, 0.9990284161960191],
]


def get_fixed_cam_pose(device: torch.device):
    """Returns (position[3], quaternion[4] w,x,y,z) in world frame for set_world_poses.

    Uses full 6-DOF from calibration (position + rotation matrix), preserves camera roll.
    """
    pos = torch.tensor(FIXED_CAM_POS, device=device)
    quat = torch.tensor(FIXED_CAM_QUAT_WXYZ, device=device)
    return pos, quat


def get_wrist_cam_offset(device: torch.device):
    """Returns (position[3], quaternion[4] w,x,y,z) offset from EE to wrist camera."""
    pos = torch.tensor(WRIST_CAM_POS, device=device)
    rot = torch.tensor(WRIST_CAM_ROT, device=device)
    quat = quat_from_matrix(rot)
    return pos, quat

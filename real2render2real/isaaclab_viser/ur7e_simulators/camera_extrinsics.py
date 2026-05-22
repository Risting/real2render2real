"""Camera extrinsics for UR7e simulation cameras.

Source: sim_data/eye_hand/calibration_result_522.json (2026-05-22, tsai, 30 valid samples)
Convention: T_A_B means p_A = T_A_B @ p_B. Camera frame is ROS/OpenCV (+Z forward, +X right, +Y down).
"""

import torch
from isaaclab.utils.math import quat_from_matrix

# --- Fixed camera: front view facing both arms and pillar (from +X direction) ---
FIXED_CAM_EYE = [1.0, 0.2, 0.6]
FIXED_CAM_TARGET = [-0.35, 0.2, 0.4]

# T_ee_cam_end: wrist camera (D405) in end-effector frame
WRIST_CAM_POS = [-0.007794556004586773, -0.06389298115559422, 0.03406491942741069]
WRIST_CAM_ROT = [
    [0.99992300522118, 0.005563967687790656, -0.011091703792162127],
    [-0.005109956798740268, 0.9991644914183526, 0.04054882772929301],
    [0.01130804894572226, -0.04048902755406822, 0.9991159926038453],
]


def get_fixed_cam_view(device: torch.device):
    """Returns (eye[3], target[3]) for set_world_poses_from_view."""
    eye = torch.tensor(FIXED_CAM_EYE, device=device)
    target = torch.tensor(FIXED_CAM_TARGET, device=device)
    return eye, target


def get_wrist_cam_offset(device: torch.device):
    """Returns (position[3], quaternion[4] w,x,y,z) offset from EE to wrist camera."""
    pos = torch.tensor(WRIST_CAM_POS, device=device)
    rot = torch.tensor(WRIST_CAM_ROT, device=device)
    quat = quat_from_matrix(rot)
    return pos, quat

"""Camera extrinsics — manual mode (user-tuned, not from calibration).

Uses the manually placed fixed camera viewpoint for quick visual debugging.
Same function signatures as camera_extrinsics.py so simulators can switch via
environment variable CAMERA_MODE=manual.
"""

import torch
from isaaclab.utils.math import quat_from_matrix

# --- Fixed camera: front view facing both arms and pillar (from +X direction) ---
# Manually tuned to look right in simulation.
FIXED_CAM_POS = [0.375, 0.125, 0.565]
FIXED_CAM_TARGET = [-0.52, 0.125, 0.12]
# Quaternion derived from eye-to-target direction with world +Z as up.
FIXED_CAM_QUAT_WXYZ = [-0.372421, 0.601085, 0.601085, -0.372421]

# Wrist camera offset: same as calibration (from calibration_result_522.json)
WRIST_CAM_POS = [-0.007794556004586773, -0.06389298115559422, 0.03406491942741069]
WRIST_CAM_ROT = [
    [0.99992300522118, 0.005563967687790656, -0.011091703792162127],
    [-0.005109956798740268, 0.9991644914183526, 0.04054882772929301],
    [0.01130804894572226, -0.04048902755406822, 0.9991159926038453],
]


def get_fixed_cam_pose(device: torch.device):
    """Returns (position[3], quaternion[4] w,x,y,z) in world frame."""
    pos = torch.tensor(FIXED_CAM_POS, device=device)
    quat = torch.tensor(FIXED_CAM_QUAT_WXYZ, device=device)
    return pos, quat


def get_wrist_cam_offset(device: torch.device):
    """Returns (position[3], quaternion[4] w,x,y,z) offset from EE to wrist camera."""
    pos = torch.tensor(WRIST_CAM_POS, device=device)
    rot = torch.tensor(WRIST_CAM_ROT, device=device)
    quat = quat_from_matrix(rot)
    return pos, quat

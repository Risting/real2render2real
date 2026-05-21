"""Camera extrinsics for UR7e simulation cameras.

Calibration source: sim_data/eye_hand/calibration_result.json (2026-05-21, tsai, 70 samples)
NOTE: calibration result is in robot base frame, ROS camera convention (+Z forward, -Y up).
      Currently using manual scene-tuned values until calibration is validated.
"""

import torch
from isaaclab.utils.math import quat_from_matrix

# --- Manual scene-tuned values (known-good viewpoints) ---
FIXED_CAM_EYE = [1.375, 1.198, 0.714]
FIXED_CAM_TARGET = [-0.4, 0.25, 0.4]
WRIST_CAM_OFFSET = [0.0, 0.0, 0.05]
WRIST_CAM_LOOK_DOWN = [0.0, 0.0, -0.2]

# --- Eye-hand calibration results (T_base_cam_fixed, robot base frame) ---
# Uncomment and switch to these once calibration is validated in sim
# CALIB_FIXED_CAM_POS = [0.1450913993555553, -0.8315226553485884, 0.3925504455925279]
# CALIB_FIXED_CAM_ROT = [
#     [-0.5175184151204286, -0.5246337213624187, -0.675969043981054],
#     [-0.44964045676549624, -0.5053865307396261, 0.736483478556499],
#     [-0.7280097181020015, 0.6850867922799163, 0.025650290303602885],
# ]


def get_fixed_cam_view(device: torch.device):
    """Returns (eye[3], target[3]) for set_world_poses_from_view."""
    eye = torch.tensor(FIXED_CAM_EYE, device=device)
    target = torch.tensor(FIXED_CAM_TARGET, device=device)
    return eye, target


def get_wrist_cam_view(ee_pos: torch.Tensor, device: torch.device):
    """Returns (eye[3], target[3]) for wrist camera given EE position."""
    offset = torch.tensor(WRIST_CAM_OFFSET, device=device)
    look = torch.tensor(WRIST_CAM_LOOK_DOWN, device=device)
    eye = ee_pos + offset
    target = ee_pos + look
    return eye, target

"""Fixed stereo camera extrinsics for spatial perception data collection.

Two cameras looking at the workspace centre (-0.4, 0.15, 0.2) from a ~0.4m baseline.
Uses ROS convention: X=right, Y=down, Z=forward.
"""

import numpy as np
import torch
from isaaclab.utils.math import quat_from_matrix


LOOK_AT = np.array([-0.50, 0.25, 0.45], dtype=np.float64)

# Two cameras at 90° to each other, Z = 1.2
CAM0_POS = np.array([0.60, -1.08, 0.85], dtype=np.float64)
CAM1_POS = np.array([0.60,  1.58, 0.85], dtype=np.float64)


def _look_at_quat(eye, target, world_up=(0.0, 0.0, 1.0)):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    wup = np.array(world_up, dtype=np.float64)
    if abs(np.dot(forward, wup)) > 0.999:
        wup = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(wup, forward)
    right = right / np.linalg.norm(right)

    down = np.cross(forward, right)

    # ROS camera-to-world: columns = [right, down, forward]
    R = np.column_stack([right, down, forward])
    return quat_from_matrix(torch.from_numpy(R).to(torch.float32))


def get_fixed_cam_poses(device: torch.device):
    quat0 = _look_at_quat(CAM0_POS, LOOK_AT)
    quat1 = _look_at_quat(CAM1_POS, LOOK_AT)

    positions = torch.from_numpy(np.stack([CAM0_POS, CAM1_POS])).to(
        device=device, dtype=torch.float32
    )
    orientations = torch.stack([quat0, quat1], dim=0).to(device)
    return positions, orientations

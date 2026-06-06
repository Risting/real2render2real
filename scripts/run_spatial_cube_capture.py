"""Launch spatial perception cube capture on the cloud server.

Usage:
    cd /root/gpufree-data/r2r2r/dependencies/IsaacLab

    # Single fixed arm pose
    ./isaaclab.sh -p ...run_spatial_cube_capture.py --num_samples 2000

    # With arm pose randomization (100 groups)
    ./isaaclab.sh -p ...run_spatial_cube_capture.py --num_samples 10000 \
        --arm_poses /root/gpufree-data/r2r2r/arm_poses.json --cubes_per_pose 100
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Generate spatial perception dataset."
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_samples", type=int, default=2000)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--arm_poses", type=str, default=None,
                    help="Path to arm_poses.json")
parser.add_argument("--cubes_per_pose", type=int, default=None,
                    help="Cube placements per arm pose group")
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from real2render2real.isaaclab_viser.configs.scene_configs.spatial_perception_cfg import (
    SpatialPerceptionCfg,
)
from real2render2real.isaaclab_viser.spatial_perception.cube_capture import (
    SpatialCubeCapture,
)


def main():
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        output_dir = os.path.join(script_dir, "..", "..", "spatial_perception_dataset")

    output_dir = os.path.abspath(output_dir)
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Number of samples: {args_cli.num_samples}")
    print(f"[INFO] Arm poses: {args_cli.arm_poses or '(default)'}")

    scene_config = SpatialPerceptionCfg(num_envs=1, env_spacing=2.0)

    capturer = SpatialCubeCapture(
        simulation_app,
        scene_config,
        output_dir=output_dir,
        num_samples=args_cli.num_samples,
        arm_poses_path=args_cli.arm_poses,
        cubes_per_pose=args_cli.cubes_per_pose,
    )
    capturer.run()
    capturer.close()


if __name__ == "__main__":
    main()

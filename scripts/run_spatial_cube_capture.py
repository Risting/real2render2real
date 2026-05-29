"""Launch spatial perception cube capture on the cloud server.

Usage:
    cd /root/gpufree-data/r2r2r/dependencies/IsaacLab
    ./isaaclab.sh -p /root/gpufree-data/r2r2r/real2render2real/scripts/run_spatial_cube_capture.py

Options:
    --num_samples N     Number of samples to generate (default: 2000)
    --output_dir DIR    Output directory (default: ../spatial_perception_dataset)
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Generate spatial perception dataset — cube images + world coords."
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_samples", type=int, default=2000,
                    help="Number of random cube placements to capture.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for the dataset.")
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
        # script_dir = .../real2render2real/scripts → ../../ = project root
        output_dir = os.path.join(script_dir, "..", "..", "spatial_perception_dataset")

    output_dir = os.path.abspath(output_dir)
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Number of samples: {args_cli.num_samples}")

    scene_config = SpatialPerceptionCfg(num_envs=1, env_spacing=2.0)

    capturer = SpatialCubeCapture(
        simulation_app,
        scene_config,
        output_dir=output_dir,
        num_samples=args_cli.num_samples,
    )
    capturer.run()
    capturer.close()


if __name__ == "__main__":
    main()

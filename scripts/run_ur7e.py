"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Run UR7e chili pick simulation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
print(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from real2render2real.isaaclab_viser.configs.scene_configs.ur7e_scene_cfg import UR7eChiliPickCfg
from real2render2real.isaaclab_viser.ur7e_simulators.ur7e_chili_pick import ChiliPick

import os
from pathlib import Path


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, "../data")
    output_data_dir = os.path.join(dir_path, "../output_data")

    scene_config = UR7eChiliPickCfg(num_envs=1, env_spacing=2.0)

    output_dir = os.path.join(output_data_dir, "ur7e_chili_pick")

    # URDF for JaxMP IK controller
    urdf_path = {
        'robot': Path(f'{data_dir}/ur5e_robotiq/ur5e.urdf'),
    }

    ChiliPick(
        simulation_app,
        scene_config,
        urdf_path=urdf_path,
        init_viser=True,
        save_data=True,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

import argparse
import os

import highway_env

from controllers import MPCController, RandomController
from sampler import ControlledSampler
from toolbox.config import Config
from toolbox.model_setup import create_dynamics_model, create_reward_model
from toolbox.utils import (append_pickle, load_checkpoint, load_config,
                           pickle_data, setup_env_with_config)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Collect MPC Controlled data.")
    parser.add_argument("-config_path", default="configs/config.json",
                        type=str, help="Path to the configuration file.")
    parser.add_argument("-env_config_path", default="configs/env_config.json",
                        type=str, help="Path to the environment configuration file.")
    parser.add_argument("-append", default=False,
                        type=str, help="Whether to append an existing data file.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        config_data = load_config(args.config_path)
        env_config_data = load_config(args.env_config_path)
    except IOError as e:
        print(f"Error reading config file: {e}")
        return

    config = Config(**config_data)
    env = setup_env_with_config(
        config.env_name, env_config_data, seed=config.seed)

    if config.controlled:
        model, optimizer, loss = create_dynamics_model(env, config)
        dynamics_model, _ = load_checkpoint(model, optimizer, os.path.join(
            config.paths.checkpoint_dir, config.paths.dynamics_file))
        dynamics_model.eval()

        model, optimizer, loss = create_reward_model(env, config)
        reward_model, _ = load_checkpoint(model, optimizer, os.path.join(
            config.paths.checkpoint_dir, config.paths.reward_file))
        reward_model.eval()

        controller = MPCController(env, reward_model, dynamics_model)
    else:
        controller = RandomController(env)

    runner = ControlledSampler(controller=controller)
    transition_data = runner.sample_transitions(
        env, config.num_episodes, verbose_num=1)

    if not os.path.exists(config.paths.data_dir):
        os.makedirs(config.paths.data_dir)

    if args.append:
        append_pickle(transition_data, config.paths.base_data_path)
    else:
        pickle_data(transition_data, config.paths.base_data_path)


if __name__ == "__main__":
    main()

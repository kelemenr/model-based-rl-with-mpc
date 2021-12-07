import argparse
import os
from datetime import datetime

import highway_env

from configs.env_configs import speed_range_env_config
from controllers import MPCController
from logger import TensorBoardLogger
from sampler import ControlledSampler
from toolbox.config import Config
from toolbox.model_setup import create_dynamics_model, create_reward_model
from toolbox.utils import (load_checkpoint, load_config, set_all_seed,
                           setup_env_with_config)

SEED = 36


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Collect MPC Controlled data.")
    parser.add_argument("-config_path", default="configs/config.json",
                        type=str, help="Path to the configuration file.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        config_data = load_config(args.config_path)
    except IOError as e:
        print(f"Error reading config file: {e}")

    config = Config(**config_data)
    env = setup_env_with_config(
        config.env_name, speed_range_env_config, seed=SEED)

    set_all_seed(SEED)

    model, optimizer, loss = create_dynamics_model(env, config)
    dynamics_model, dynamics_optimizer = load_checkpoint(model, optimizer, os.path.join(
        config.paths.checkpoint_dir, config.paths.dynamics_file))
    dynamics_model.eval()

    model, optimizer, loss = create_reward_model(env, config)
    reward_model, reward_optimizer = load_checkpoint(model, optimizer, os.path.join(
        config.paths.checkpoint_dir, config.paths.reward_file))
    reward_model.eval()

    logger = TensorBoardLogger(
        f"{config.paths.logger_dir}/env_{config.env_name}_epochs_{config.model.num_epochs}_rhidden_"
        f"{config.model.reward_hidden_size}_dhidden_{config.model.dynamics_hidden_size}_lr_"
        f"{config.model.learning_rate}_batch_{config.model.batch_size}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    mpc = MPCController(env, reward_model, dynamics_model)
    runner = ControlledSampler(controller=mpc, logger=logger)
    runner.sample_transitions(env, config.num_episodes, verbose_num=1)


if __name__ == "__main__":
    main()

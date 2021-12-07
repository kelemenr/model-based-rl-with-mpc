import argparse
import os
import random

import highway_env
from torch.optim.lr_scheduler import ExponentialLR

from configs.env_configs import speed_range_env_config
from controllers import MPCController
from dataloader import TransitionData
from sampler import ControlledSampler
from toolbox.config import Config
from toolbox.model_setup import (create_dynamics_model, create_reward_model,
                                 setup_model)
from toolbox.utils import (load_checkpoint, load_config, read_pickle,
                           set_all_seed, setup_env_with_config,
                           train_test_split)
from trainer import ModelTrainer

SEED = 36


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Collect MPC Controlled data.")
    parser.add_argument("-config_path", default="configs/config.json",
                        type=str, help="Path to the configuration file.")
    parser.add_argument("-visualize", default=False,
                        type=bool, help="Visualize training.")
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

    dataset = read_pickle(config.paths.base_data_path)

    model, optimizer, loss = create_dynamics_model(env, config)
    dynamics_model, dynamics_optimizer = load_checkpoint(model, optimizer, os.path.join(
        config.paths.checkpoint_dir, config.paths.dynamics_file))
    dynamics_model.eval()

    model, optimizer, loss = create_reward_model(env, config)
    reward_model, reward_optimizer = load_checkpoint(model, optimizer, os.path.join(
        config.paths.checkpoint_dir, config.paths.reward_file))
    reward_model.eval()

    mpc = MPCController(env, reward_model, dynamics_model)
    for iteration in range(config.num_iterations):
        runner = ControlledSampler(controller=mpc)
        controlled_data = runner.sample_transitions(env,
                                                    config.num_episodes,
                                                    verbose_num=1)

        controlled_data = dataset + controlled_data

        train_controlled, validation_controlled = train_test_split(
            controlled_data,
            config.model.train_split_ratio)

        controlled_train_data = TransitionData(train_controlled)
        controlled_validation_data = TransitionData(validation_controlled)

        model_config = setup_model(config.model.model_type)
        model, optimizer, loss = model_config(env, config)
        retrain_model, optimizer = load_checkpoint(
            model, optimizer, config.paths.checkpoint_path)

        retrain_model.train()

        trainer = ModelTrainer(retrain_model, optimizer, loss, ExponentialLR(
            optimizer, gamma=config.model.scheduler_gamma), config.device, config.paths.logger_dir, config.paths.checkpoint_path)

        trainer.train(controlled_train_data, controlled_validation_data,
                      config.model.num_epochs, config.model.batch_size, visualize=args.visualize)

        retrain_model.eval()

        mpc = MPCController(env, reward_model, retrain_model)


if __name__ == "__main__":
    main()

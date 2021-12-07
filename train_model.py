import argparse
import os

import highway_env
from torch.optim.lr_scheduler import ExponentialLR

from configs.env_configs import speed_range_env_config
from dataloader import TransitionData
from toolbox.config import Config
from toolbox.model_setup import setup_model
from toolbox.utils import (load_config, read_pickle, save_model, set_all_seed,
                           setup_env_with_config, train_test_split)
from trainer import ModelTrainer

SEED = 42


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train model with transition data.")
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
        return

    config = Config(**config_data)
    env = setup_env_with_config(
        config.env_name, speed_range_env_config, seed=SEED)

    set_all_seed(SEED)

    dataset = read_pickle(config.paths.base_data_path)

    train_dataset, validation_dataset = train_test_split(
        dataset, config.model.train_split_ratio)

    train_data = TransitionData(train_dataset)
    validation_data = TransitionData(validation_dataset)

    model_config = setup_model(config.model.model_type)
    model, optimizer, loss = model_config(env, config)

    trainer = ModelTrainer(model, optimizer, loss, ExponentialLR(
        optimizer, gamma=config.model.scheduler_gamma), config.device, config.paths.logger_dir, config.paths.checkpoint_path)

    trainer.train(train_data, validation_data,
                  config.model.num_epochs, config.model.batch_size, visualize=args.visualize)

    if not os.path.exists(config.paths.model_dir):
        os.makedirs(config.paths.model_dir)

    save_model(model, config.paths.model_path)


if __name__ == "__main__":
    main()

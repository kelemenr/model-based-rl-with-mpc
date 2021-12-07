import json
import pickle
import random

import gym
import numpy as np
import torch

# from sampler import Transition


def train_test_split(dataset, train_ratio):
    train_data, test_data = dataset[:int(train_ratio * len(dataset))], \
        dataset[int(train_ratio * len(dataset)):]
    return train_data, test_data


def setup_env_with_config(env_name, config, seed):
    env = gym.make(env_name)
    env.seed(seed)

    env.configure(config)
    env.reset()
    return env


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model


# def transpose_batch(batch):
#    return Transition(*map(torch.stack, zip(*batch)))


def pickle_data(data, file_path):
    with open(file_path, "wb") as dataset_pickle:
        pickle.dump(data, dataset_pickle)


def read_pickle(file_path):
    with open(file_path, "rb") as dataset_pickle:
        dataset = pickle.load(dataset_pickle)
    return dataset


def read_all_pickle(file_path):
    file = open(file_path, 'rb')
    objects = []
    while 1:
        try:
            objects.append(pickle.load(file))
        except EOFError:
            break
    return


def append_pickle(data, file_path):
    pickle.dump(data, file_path)


def set_all_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gym.utils.seeding.np_random(seed)


def save_seed(val, filename):
    with open(filename, "wb") as f:
        f.write(str(val))


def load_seed(filename):
    with open(filename, "rb") as f:
        # make sure that datatype is in accordance with seed datatype
        return int(f.read())


def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_checkpoint(epoch, model, optimizer, checkpoint_path):
    state = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print(f"=> saving checkpoint to {checkpoint_path}")


def load_checkpoint(model, optimizer, filename):
    try:
        print(f"=> Attempting to load checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', None)
        print(f"=> Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return model, optimizer
    except FileNotFoundError:
        print(
            f"Checkpoint file not found: {filename}. Continuing without loading.")
        return model, optimizer


def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

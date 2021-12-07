import pickle
from abc import ABC
from collections import namedtuple
from itertools import count
from timeit import default_timer as timer

import numpy as np
import torch

Transition = namedtuple("Transition",
                        ["state", "action", "next_state", "reward"])


class Sampler(ABC):
    def __init__(self):
        pass

    def sample_transitions(self, env, num_episodes, verbose_num):
        """
        Returns transition samples.
        """
        pass


class ControlledSampler(Sampler):
    def __init__(self, controller, eval=False, logger=None):
        super().__init__()
        self.controller = controller
        self.eval = eval
        self.logger = logger

    def sample_transitions(self, env, num_episodes, verbose_num):
        """
        Interacts with the environment according to a controller.
        :param env: gym environment
        :param num_episodes: number of environment interaction episodes
        :param verbose_num: prints the episode data every nth episodes
        :return: transition data
        """
        transitions = []
        cumulated_reward = 0
        for episode in range(num_episodes):
            state = env.reset()
            state = torch.Tensor(state.flatten())

            episode_reward = 0
            episode_start = timer()
            for time_step in count():
                env.render()
                action = np.array([self.controller.select_action(state)])

                next_state, reward, done, _ = env.step(action[0])
                next_state = torch.Tensor(
                    next_state.flatten())

                episode_reward += reward

                if done and time_step != 39:
                    reward = 0.0

                transitions.append(Transition(torch.Tensor(state), torch.Tensor(
                    action), torch.Tensor(next_state), torch.Tensor(np.array(reward))))

                state = next_state

                if done:
                    episode_end = timer()
                    episode_length = round(episode_end - episode_start, 6)
                    episode_timesteps = time_step
                    break

            cumulated_reward += episode_reward

            if episode % verbose_num == 0:
                print(
                    f" Recording episode: {episode} || timesteps: "
                    f"{time_step} ||reward: {episode_reward} || length: {episode_length} ...")

            if self.logger is not None:
                self.logger.log_scalar(
                    "episode_rewards", episode_reward, episode)
                self.logger.log_scalar(
                    "episode_lengths", episode_length, episode)
                self.logger.log_scalar(
                    "episode_timesteps", episode_timesteps, episode)
                self.logger.log_scalar(
                    "cumulated_rewards", cumulated_reward, episode)
        return transitions

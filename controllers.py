import copy
from abc import ABC

import numpy as np
import torch


class Controller(ABC):
    def __init__(self):
        pass

    def select_action(self, state):
        """
        Returns an action for a given state.
        """
        pass


class RandomController(Controller):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def select_action(self, state):
        """
        Returns random action.
        :param state: current environment state
        :return: random action sampled from the environment
        """
        return self.env.action_space.sample()


class MPCController(Controller):
    def __init__(self, env, reward_model, dynamics_model):
        super().__init__()
        self.env = env
        self.reward_model = reward_model
        self.dynamics_model = dynamics_model

    def select_action(self, state):
        """
        Returns an action according to Model Predictive Control.
        :param state: current environment state
        :return: best action or action sequence
        """
        n = self.env.action_space.n
        possible_actions = torch.tensor(
            [i for i in range(n)]).unsqueeze(-1).float()
        states_list = torch.stack([state for _ in range(n)])

        # TODO make a loop.. :D

        with torch.no_grad():
            # paths from s_0: 5
            expected_rewards = self.reward_model(states_list, possible_actions).gather(
                1, possible_actions.type(torch.int64))
            expected_next_states = self.dynamics_model(
                states_list, possible_actions).detach()
            trajectory_reward = expected_rewards

            # paths from s_0: 5x5
            repeated_expected_next_states = expected_next_states.repeat_interleave(
                n, dim=0)  # same state 5 times
            repeated_actions = possible_actions.repeat(
                n, 1)  # pair each state with all 5 actions
            expected_rewards2 = self.reward_model(repeated_expected_next_states, repeated_actions).gather(
                1, repeated_actions.type(torch.int64)).split(n)  # pair each state with all 5 rewards for all 5 actions
            expected_next_states2 = self.dynamics_model(
                repeated_expected_next_states, repeated_actions).detach()

            # choose the largest reward for each possible state
            trajectory_reward += torch.FloatTensor([max(r)
                                                   for r in expected_rewards2]).reshape(n, 1)

            # paths from s_0 5x5x5
            repeated_expected_next_states3 = expected_next_states2.repeat_interleave(
                n, dim=0)  # 25 states, same state 5 times
            repeated_actions3 = possible_actions.repeat(
                n*n, 1)  # pair each state with all 5 actions
            expected_rewards3 = self.reward_model(repeated_expected_next_states3, repeated_actions3).gather(
                1, repeated_actions3.type(torch.int64)).split(n*n)
            expected_next_states3 = self.dynamics_model(
                repeated_expected_next_states3, repeated_actions3).detach()

            trajectory_reward += torch.FloatTensor([max(r)
                                                   for r in expected_rewards3]).reshape(n, 1)

            # paths from s_0 5x5x5x5
            repeated_expected_next_states4 = expected_next_states3.repeat_interleave(
                n, dim=0)
            repeated_actions4 = possible_actions.repeat(n*n*n, 1)
            expected_rewards4 = self.reward_model(repeated_expected_next_states4, repeated_actions4).gather(
                1, repeated_actions4.type(torch.int64)).split(n*n*n)

            trajectory_reward += torch.FloatTensor([max(r)
                                                   for r in expected_rewards4]).reshape(n, 1)

            best_action = torch.argmax(trajectory_reward)
        return best_action

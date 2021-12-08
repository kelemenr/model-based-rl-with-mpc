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
    def __init__(self, env, reward_model, dynamics_model, horizon=4):
        super().__init__()
        self.env = env
        self.reward_model = reward_model
        self.dynamics_model = dynamics_model
        self.horizon = horizon

    def select_action(self, state):
        """
        Returns an action according to Model Predictive Control.
        :param state: current environment state
        :return: best action or action sequence
        """
        n = self.env.action_space.n
        possible_actions = torch.arange(n).unsqueeze(-1).float()
        states_list = torch.stack([state] * n)

        with torch.no_grad():
            expected_rewards = self.reward_model(states_list, possible_actions).gather(
                1, possible_actions.type(torch.int64))
            expected_next_states = self.dynamics_model(
                states_list, possible_actions).detach()

            trajectory_reward = expected_rewards
            for i in range(1, self.horizon):
                repeated_expected_next_states = expected_next_states.repeat_interleave(
                    n, dim=0)
                repeated_actions = possible_actions.repeat(n**i, 1)

                expected_rewards = self.reward_model(repeated_expected_next_states, repeated_actions).gather(
                    1, repeated_actions.type(torch.int64)).split(n**i)
                expected_next_states = self.dynamics_model(
                    repeated_expected_next_states, repeated_actions).detach()

                trajectory_reward += torch.FloatTensor(
                    [max(r) for r in expected_rewards]).reshape(n, 1)

            best_action = torch.argmax(trajectory_reward)
        return best_action

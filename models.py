import torch
import torch.nn as nn
from torch import nn
from torch.functional import F


class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, dt):
        super().__init__()
        """
        s_t+1 = f(s_t, a_t) = A(s_t, a_t)s + B(s_t, a_t)a
        """
        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        A_size = state_size * state_size
        B_size = state_size * action_size
        self.A1 = nn.Linear(state_size + action_size, hidden_size)
        self.A2 = nn.Linear(hidden_size, A_size)
        self.B1 = nn.Linear(state_size + action_size, hidden_size)
        self.B2 = nn.Linear(hidden_size, B_size)

    def resize_input(self, state_batch, action_batch):
        if state_batch.shape[0] == self.state_size or state_batch.shape[0] == 1:
            state_batch = state_batch.reshape(1, self.state_size)
            action_batch = action_batch.reshape(1, 1)
            return torch.cat((state_batch, action_batch), -1)
        else:
            return torch.cat((state_batch, action_batch), -1)

    def forward(self, state_batch, action_batch):
        """
        Predict s_t+1 = f(s_t, a_t)
        :param state_batch: a batch of states
        :param action_batch: a batch of actions
        """
        state_action_batch = self.resize_input(state_batch, action_batch)

        A = self.A2(F.relu(self.A1(state_action_batch)))
        A = torch.reshape(A, (
            state_batch.shape[0], self.state_size, self.state_size))
        B = self.B2(F.relu(self.B1(state_action_batch)))
        B = torch.reshape(B, (
            state_batch.shape[0], self.state_size, self.action_size))

        ds = A @ state_batch.unsqueeze(-1).float() + \
            B @ action_batch.unsqueeze(-1).float()
        return state_batch + ds.squeeze() * self.dt

    @staticmethod
    def get_loss(model, transition_data, loss_function, device):
        states, actions, next_states, rewards = transition_data
        predictions = model(states, actions)
        return loss_function(predictions, next_states)


class DynamicsModelLSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, dt, num_layers=2):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=state_size + action_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc_A = nn.Linear(hidden_size, state_size * state_size)
        self.fc_B = nn.Linear(hidden_size, state_size * action_size)

    def forward(self, state_sequence, action_sequence):
        state_action_sequence = torch.cat(
            (state_sequence, action_sequence), -1)
        lstm_out, _ = self.lstm(state_action_sequence)
        A = torch.reshape(self.fc_A(lstm_out),
                          (-1, self.state_size, self.state_size))
        B = torch.reshape(self.fc_B(lstm_out),
                          (-1, self.state_size, self.action_size))
        ds = A @ state_sequence.unsqueeze(-1).float() + \
            B @ action_sequence.unsqueeze(-1).float()
        return state_sequence + ds.squeeze() * self.dt

    @staticmethod
    def get_loss(model, transition_data, loss_function, device):
        states, actions, next_states, rewards = transition_data
        predictions = model(states, actions)
        return loss_function(predictions, next_states)


class RewardModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(RewardModel, self).__init__()
        self.state_size = state_size

        self.lin1 = nn.Linear(state_size + 1, hidden_size // 2)
        self.lin2 = nn.Linear(hidden_size // 2, hidden_size)
        self.d1 = nn.Dropout(0.75)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.d2 = nn.Dropout(0.75)
        self.lin4 = nn.Linear(hidden_size, hidden_size)

        linear_input_size = hidden_size
        self.head = nn.Linear(linear_input_size, 5)
        self.relu = nn.ReLU()

    def resize_input(self, state_batch, action_batch):
        if state_batch.shape[0] == self.state_size or state_batch.shape[0] == 1:
            state_batch = state_batch.reshape(1, self.state_size)
            action_batch = action_batch.reshape(1, 1)
            return torch.cat((state_batch, action_batch), -1)
        else:
            return torch.cat((state_batch, action_batch), -1)

    def forward(self, state_batch, action_batch):
        x = self.resize_input(state_batch, action_batch)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.d1(x)
        x = self.relu(self.lin3(x))
        x = self.d2(x)
        x = self.relu(self.lin4(x))
        return self.head(x.view(x.size(0), -1))

    @staticmethod
    def get_loss(model, transition_data, loss_function, device):
        states, actions, next_states, rewards = transition_data
        predictions = model(states, actions).gather(
            1, actions.type(torch.int64))
        return loss_function(predictions,
                             rewards.reshape(rewards.shape[0], 1))


class RewardModelLSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(RewardModelLSTM, self).__init__()
        self.state_size = state_size

        self.lstm = nn.LSTM(input_size=state_size + 1,
                            hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.lin = nn.Linear(hidden_size, 5)
        self.relu = nn.ReLU()

    def resize_input(self, state_batch, action_batch):
        if state_batch.shape[0] == self.state_size or state_batch.shape[0] == 1:
            state_batch = state_batch.reshape(1, self.state_size)
            action_batch = action_batch.reshape(1, 1)
            return torch.cat((state_batch, action_batch), -1)
        else:
            return torch.cat((state_batch, action_batch), -1)

    def forward(self, state_batch, action_batch):
        x = self.resize_input(state_batch, action_batch)
        lstm_out, _ = self.lstm(x)
        x = self.relu(lstm_out)
        return self.lin(x)

    @staticmethod
    def get_loss(model, transition_data, loss_function, device):
        states, actions, next_states, rewards = transition_data
        predictions = model(states, actions).gather(
            1, actions.type(torch.int64))
        return loss_function(predictions, rewards.reshape(rewards.shape[0], 1))

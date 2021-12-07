import os

import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import TensorBoardLogger
from toolbox.utils import save_checkpoint
from toolbox.visualize import visualize_train_valid_loss


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


def train(model, train_data, validation_data,
          optimizer, loss_function, epochs, batch_size,
          device, checkpoint_path, visualize, scheduler):

    logger = TensorBoardLogger(f"./logs/run/highway__mbrl1")

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = DataLoader(validation_data, batch_size, shuffle=False)

    train_valid_loss = np.full((epochs, 2), np.nan)
    for epoch in tqdm(range(epochs), desc="Training model...", mininterval=15):
        running_loss = 0.0
        running_val_loss = 0.0
        model.train()
        for train_inputs in train_loader:
            optimizer.zero_grad()
            train_loss = model.get_loss(model, train_inputs, loss_function,
                                        device)
            train_loss.backward()
            for param in model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            running_loss += train_loss.item()
        training_loss = running_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            for valid_inputs in valid_loader:
                valid_loss = model.get_loss(model, valid_inputs,
                                            loss_function, device)
                running_val_loss += valid_loss.item()
        validation_loss = running_val_loss / len(valid_loader)

        if epoch > 0:
            if epoch % 1 == 0:
                scheduler.step()
                print(
                    f"=> Scheduler: learning rate is set to {optimizer.param_groups[0]['lr']}")
                torch.save(model.state_dict(), "temp_model.pt")

        print(f"Epoch {epoch}:\t training loss: {training_loss}\t "
              f"validation loss: {validation_loss}")

        logger.log_scalar("training_loss", training_loss, epoch)
        logger.log_scalar("validation_loss", validation_loss, epoch)

        train_valid_loss[epoch] = [training_loss, validation_loss]

    save_checkpoint(epochs, model, optimizer, checkpoint_path)

    if visualize:
        visualize_train_valid_loss(train_valid_loss)
    return model, optimizer

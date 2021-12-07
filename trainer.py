import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import TensorBoardLogger
from toolbox.utils import save_checkpoint
from toolbox.visualize import visualize_train_valid_loss


class ModelTrainer:
    def __init__(self, model, optimizer, loss_function, scheduler, device, logger_path, checkpoint_path):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.device = device
        self.logger = TensorBoardLogger(logger_path)
        self.checkpoint_path = checkpoint_path

    def train(self, train_data, validation_data, epochs, batch_size, visualize):
        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        valid_loader = DataLoader(validation_data, batch_size, shuffle=False)

        train_valid_loss = np.full((epochs, 2), np.nan)

        for epoch in tqdm(range(epochs), desc="Training model...", mininterval=5):
            print(f"Epoch {epoch}", end="\n")
            self._train_epoch(train_loader, epoch)
            self._validate_epoch(valid_loader, epoch)

            if visualize:
                visualize_train_valid_loss(train_valid_loss)

        save_checkpoint(epochs, self.model, self.optimizer,
                        self.checkpoint_path)
        return self.model, self.optimizer

    def _train_epoch(self, train_loader, epoch):
        running_loss = 0.0
        self.model.train()
        for train_inputs in train_loader:
            self.optimizer.zero_grad()
            train_loss = self.model.get_loss(
                self.model, train_inputs, self.loss_function, self.device)
            train_loss.backward()
            self._clip_gradients()
            self.optimizer.step()
            running_loss += train_loss.item()

        training_loss = running_loss / len(train_loader)
        print(f"training_loss: {round(training_loss, 5)}", end=" ")
        self.logger.log_scalar("training_loss", training_loss, epoch)

    def _validate_epoch(self, valid_loader, epoch):
        running_val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for valid_inputs in valid_loader:
                valid_loss = self.model.get_loss(
                    self.model, valid_inputs, self.loss_function, self.device)
                running_val_loss += valid_loss.item()

        validation_loss = running_val_loss / len(valid_loader)
        print(f"validation_loss: {round(validation_loss, 5)}")
        self.logger.log_scalar("validation_loss", validation_loss, epoch)

    def _clip_gradients(self, clip_value=1.0):
        for param in self.model.parameters():
            param.grad.data.clamp_(-clip_value, clip_value)

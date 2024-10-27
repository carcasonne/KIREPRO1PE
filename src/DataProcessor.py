from typing import TypeVar

import torch
from torch import nn
from torch.nn import Module as TorchLoss  # this is the base class for all criteria
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ClonedAudioDetector import CNNClassifier

CType = TypeVar('CType', bound=TorchLoss)
OType = TypeVar('OType', bound=Optimizer)


class DataProcessor():
    def __init__(self, model: CNNClassifier, use_coda: bool):
        self.model = model
        self.use_coda = use_coda

    def process(self,
                training_data: DataLoader,
                validation_data: DataLoader,
                no_of_epochs):
        """trains a cnn classifier on given data and returns results"""
        criterion = nn.CrossEntropyLoss()
        optimizer = Optimizer.Adadelta(self.model.parameters())
        return self.train_model(self.model, training_data, validation_data, no_of_epochs, criterion, optimizer, self.use_coda)

    def train_model(self,
                    model: CNNClassifier,
                    training_data_loader: DataLoader,
                    validation_data_loader: DataLoader,
                    no_of_epochs: int,
                    criterion: CType,
                    optimizer: OType,
                    use_coda_if_available: bool):
        """trains a cnn classifier on given data and returns results"""
        device = "cpu"
        if use_coda_if_available and torch.cuda.is_available():
            device = "cuda"

        for epoch in range(no_of_epochs):
            # Training
            model.train()
            running_loss = 0.0
            for images, labels in training_data_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                print(images.shape)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            self.print_training_result_epoch(
                epoch + 1, no_of_epochs, running_loss, len(training_data_loader.dataset))

            # Validation
            model.eval()
            val_loss, correct = 0.0, 0
            with torch.no_grad():
                for images, labels in validation_data_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()

            self.print_validation_result_epoch(
                val_loss, correct, len(validation_data_loader.dataset))

    def print_training_result_epoch(self, epoch, no_of_epochs, running_loss, data_size):
        print(f"Epoch [{epoch}/{no_of_epochs}], Training Loss: {
            running_loss / data_size:.4f}")

    def print_validation_result_epoch(self, val_loss, correct, data_size):
        print(f"Validation Loss: {val_loss / data_size:.4f}, Accuracy: {
            correct / data_size:.4f}")

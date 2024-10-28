from typing import TypeVar

import torch
from torch import nn
from torch.nn import Module as TorchLoss  # this is the base class for all criteria
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from Application.ClonedAudioDetector import CNNClassifier
from Core.Metrics import EpochMetrics, MetricType, Results

CType = TypeVar("CType", bound=TorchLoss)
OType = TypeVar("OType", bound=Optimizer)


class DataProcessor:
    def __init__(self, model: nn.Module, use_cuda: bool):
        self.model = model
        self.use_cuda = use_cuda

    def process(self, training_data: DataLoader, validation_data: DataLoader, n_epochs) -> Results:
        """processes the data by training the cnnclassifier with it
        this WILL affect the model associated with this DataProcessor
        return results from training"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adadelta(self.model.parameters())
        return self.train_model(
            self.model,
            training_data,
            validation_data,
            n_epochs,
            criterion,
            optimizer,
            self.use_cuda,
        )

    def train_model(
        self,
        model: CNNClassifier,
        training_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        n_epochs: int,
        criterion: CType,
        optimizer: OType,
        use_cuda_if_available: bool,
    ) -> Results:
        """trains a cnn classifier on given data and returns results"""
        device = "cpu"
        if use_cuda_if_available and torch.cuda.is_available():
            device = "cuda"

        results = Results()

        for epoch in range(n_epochs):
            print(f"Beginning training epoch no. {epoch + 1}")

            # Training
            model.train()
            training_loss, training_correct = 0.0, 0
            for images, labels in training_data_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                # print(images.shape)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                training_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                training_correct += (predicted == labels).sum().item()

            training_metrics = self.get_training_metrics(
                training_loss, training_correct, len(training_data_loader.dataset)
            )

            # Validation
            model.eval()
            validation_loss, validation_correct = 0.0, 0
            with torch.no_grad():
                for images, labels in validation_data_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    validation_correct += (predicted == labels).sum().item()

            validation_metrics = self.get_validation_metrics(
                validation_loss, validation_correct, len(validation_data_loader.dataset)
            )

            results.training.append(training_metrics)
            results.validation.append(validation_metrics)

        return results

    def get_training_metrics(
        self, absolute_loss: float, correct: int, data_size: float
    ) -> EpochMetrics:
        training_metrics = EpochMetrics()
        training_metrics.add_metric(MetricType.ABSOLUTE_LOSS, absolute_loss)
        training_metrics.add_metric(MetricType.AVERAGE_LOSS, absolute_loss / data_size)
        training_metrics.add_metric(MetricType.ACCURACY, correct / data_size)
        return training_metrics

    def get_validation_metrics(
        self, val_loss: float, correct: int, data_size: float
    ) -> EpochMetrics:
        validation_metrics = EpochMetrics()
        validation_metrics.add_metric(MetricType.ABSOLUTE_LOSS, val_loss)
        validation_metrics.add_metric(MetricType.AVERAGE_LOSS, val_loss / data_size)
        validation_metrics.add_metric(MetricType.ACCURACY, correct / data_size)
        return validation_metrics

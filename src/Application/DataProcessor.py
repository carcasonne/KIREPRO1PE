import json
import time
from typing import List, TypeVar

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score
from sklearn.model_selection import KFold
from torch import nn
from torch.nn import Module as TorchLoss  # this is the base class for all criteria
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

from Application.ClonedAudioDetector import CNNClassifier
from Core.Metrics import EpochMetrics, MetricType, Results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


CType = TypeVar("CType", bound=TorchLoss)
OType = TypeVar("OType", bound=Optimizer)


class DataProcessor:
    def __init__(self, model: nn.Module, use_cuda: bool):
        self.model = model
        self.use_cuda = use_cuda

    def process(
        self,
        model_class,
        training_data: DataLoader,
        validation_data: DataLoader,
        n_epochs,
        learning_rate,
    ) -> tuple[Results, float]:
        """processes the data by training the cnnclassifier with it
        this WILL affect the model associated with this DataProcessor
        return results from training"""
        criterion = nn.CrossEntropyLoss()

        start = time.perf_counter()

        # Create fresh model for each fold
        model = model_class(3)
        # Create optimizer for THIS model's parameters
        optimizer = torch.optim.Adadelta(model.parameters())
        fold_results = self.train_model(
            self.model,
            training_data,
            validation_data,
            n_epochs,
            criterion,
            optimizer,
            self.use_cuda,
        )

        end = time.perf_counter()
        complete_time = end - start

        return fold_results, complete_time

    def process_k_fold(
        self,
        model_class,
        dataset: DataLoader,
        k_folds: int,
        n_epochs: int,
        batch_size: int,
        use_cuda_if_available: bool,
    ) -> tuple[List[Results], float]:
        kfold = KFold(n_splits=k_folds, shuffle=True)
        fold_results = []

        start = time.perf_counter()

        for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
            print(f"Starting fold {fold + 1}/{k_folds}")

            # Create fresh model for each fold
            model = model_class(3)
            # Create optimizer for THIS model's parameters
            optimizer = torch.optim.Adadelta(model.parameters())  # not self.model.parameters()
            criterion = nn.CrossEntropyLoss()

            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

            results = self.train_model(
                model=model,  # pass the new model
                training_data_loader=train_loader,
                validation_data_loader=val_loader,
                n_epochs=n_epochs,
                criterion=criterion,
                optimizer=optimizer,
                use_cuda_if_available=use_cuda_if_available,
            )

            fold_results.append(results)

        end = time.perf_counter()
        complete_time = end - start

        return fold_results, complete_time

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

        model = model.to(device)
        results = Results()

        for epoch in range(n_epochs):
            print(f"Beginning training epoch no. {epoch + 1}")

            # Trainingc
            model.train()
            training_loss = 0.0
            all_training_preds = []
            all_training_labels = []

            for batch_idx, (images, labels) in enumerate(training_data_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                training_loss += loss.item() * images.size(0)

                all_training_preds.extend(predicted.cpu())
                all_training_labels.extend(labels.cpu())

            all_training_preds = torch.stack(all_training_preds)  # convert lists to tensors
            all_training_labels = torch.stack(all_training_labels)
            training_metrics = self.get_training_metrics(
                training_loss,
                all_training_preds,
                all_training_labels,
                len(training_data_loader.dataset),
            )
            results.training.append(training_metrics)

            # Validation
            model.eval()
            validation_loss = 0.0
            all_validation_preds = []
            all_validation_labels = []
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(validation_data_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item() * images.size(0)

                    _, predicted = torch.max(outputs, 1)
                    all_validation_preds.extend(predicted.cpu())
                    all_validation_labels.extend(labels.cpu())

            all_validation_preds = torch.stack(all_validation_preds)
            all_validation_labels = torch.stack(all_validation_labels)
            validation_metrics = self.get_validation_metrics(
                validation_loss,
                all_validation_preds,
                all_validation_labels,
                len(validation_data_loader.dataset),
            )
            results.validation.append(validation_metrics)

        return results

    def get_training_metrics(
        self, absolute_loss: float, predicted: torch.Tensor, labels: torch.Tensor, data_size: float
    ) -> EpochMetrics:
        training_metrics = EpochMetrics()
        # move tensors to cpu and convert to numpy
        y_true = labels.cpu().numpy()
        y_pred = predicted.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        training_metrics.add_metric(MetricType.AVERAGE_LOSS, absolute_loss / data_size)
        training_metrics.add_metric(MetricType.ACCURACY, (tp + tn) / data_size)
        training_metrics.add_metric(
            MetricType.PRECISION, precision_score(y_true, y_pred, zero_division=0)
        )
        training_metrics.add_metric(MetricType.F1_SCORE, f1_score(y_true, y_pred, zero_division=0))

        # confusion matrix stuff
        training_metrics.add_metric(MetricType.TRUE_POSITIVES, int(tp))
        training_metrics.add_metric(MetricType.TRUE_NEGATIVES, int(tn))
        training_metrics.add_metric(MetricType.FALSE_POSITIVES, int(fp))
        training_metrics.add_metric(MetricType.FALSE_NEGATIVES, int(fn))

        return training_metrics

    def get_validation_metrics(
        self, val_loss: float, predicted: torch.Tensor, labels: torch.Tensor, data_size: float
    ) -> EpochMetrics:
        validation_metrics = EpochMetrics()
        # move tensors to cpu and convert to numpy
        y_true = labels.cpu().numpy()
        y_pred = predicted.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        validation_metrics.add_metric(MetricType.AVERAGE_LOSS, val_loss / data_size)
        validation_metrics.add_metric(MetricType.ACCURACY, (tp + tn) / data_size)
        validation_metrics.add_metric(
            MetricType.PRECISION, precision_score(y_true, y_pred, zero_division=0)
        )
        validation_metrics.add_metric(
            MetricType.F1_SCORE, f1_score(y_true, y_pred, zero_division=0)
        )

        # confusion matrix stuff
        validation_metrics.add_metric(MetricType.TRUE_POSITIVES, int(tp))
        validation_metrics.add_metric(MetricType.TRUE_NEGATIVES, int(tn))
        validation_metrics.add_metric(MetricType.FALSE_POSITIVES, int(fp))
        validation_metrics.add_metric(MetricType.FALSE_NEGATIVES, int(fn))

        return validation_metrics

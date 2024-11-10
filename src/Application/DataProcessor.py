import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, TypeVar

import numpy as np
import torch
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
        self, training_data: DataLoader, validation_data: DataLoader, n_epochs, learning_rate
    ) -> tuple[Results, float]:
        """processes the data by training the cnnclassifier with it
        this WILL affect the model associated with this DataProcessor
        return results from training"""
        criterion = nn.CrossEntropyLoss()

        start = time.perf_counter()

        optimizer = torch.optim.Adadelta(self.model.parameters())
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

        print(f"Training on device: {device}")

        debug_stats = {
            "train_output_dist": [],
            "val_output_dist": [],
            "gradient_norms": [],
            "layer_activations": [],
        }

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")

            # Training phase
            model.train()
            training_loss, training_correct = 0.0, 0
            batch_outputs = []

            for batch_idx, (images, labels) in enumerate(training_data_loader):
                images, labels = images.to(device), labels.to(device)

                print(f"Batch {batch_idx}: input range: [{images.min():.3f}, {images.max():.3f}]")

                optimizer.zero_grad()
                outputs, layer_debug = model(images)
                batch_outputs.append(outputs.detach().cpu().numpy())

                if batch_idx == 0:  # print debug info for first batch
                    print(f"Layer activations: {layer_debug}")

                pred_dist = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                print(
                    f"Prediction distribution: mean={pred_dist.mean():.3f}, std={pred_dist.std():.3f}"
                )

                loss = criterion(outputs, labels)
                loss.backward()

                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5
                debug_stats["gradient_norms"].append(total_norm)

                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx} gradient norm: {total_norm:.3f}")

                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                training_correct += (predicted == labels).sum().item()
                training_loss += loss.item() * images.size(0)

            print("\nSwitching to eval mode...")
            model.eval()
            print(f"Model training? {model.training}")
            validation_loss, validation_correct = 0.0, 0
            val_outputs = []
            prev_batch_preds = None

            print("\nStarting validation...")
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(validation_data_loader):
                    images, labels = images.to(device), labels.to(device)

                    # Handle both possible return types
                    out = model(images)
                    if isinstance(out, tuple):
                        outputs, layer_debug = out
                        if batch_idx == 0:
                            print("WARNING: Model returning debug info in eval mode!")
                            print(f"Layer debug: {layer_debug}")
                    else:
                        outputs = out

                    current_preds = torch.max(outputs, 1)[1].cpu().numpy()
                    val_outputs.append(outputs.detach().cpu().numpy())

                    val_pred_dist = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                    print(
                        f"Val batch {batch_idx} pred distribution: mean={val_pred_dist.mean():.3f}, std={val_pred_dist.std():.3f}"
                    )

                    # check if predictions are identical within THIS batch
                    if len(np.unique(current_preds)) == 1:
                        print(
                            f"WARNING: All predictions in batch {batch_idx} are identical: {current_preds[0]}"
                        )

                    # check if this batch's predictions match previous batch's predictions
                    if prev_batch_preds is not None:
                        matches = np.mean(
                            current_preds[: min(len(current_preds), len(prev_batch_preds))]
                            == prev_batch_preds[: min(len(current_preds), len(prev_batch_preds))]
                        )
                        if matches > 0.99:  # allow for small differences
                            print(
                                f"WARNING: {matches*100:.1f}% identical predictions between batches!"
                            )

                    prev_batch_preds = current_preds

                    loss = criterion(outputs, labels)
                    validation_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    validation_correct += (predicted == labels).sum().item()

            # Calculate metrics
            train_size = len(training_data_loader.dataset)
            val_size = len(validation_data_loader.dataset)

            training_metrics = self.get_training_metrics(
                training_loss, training_correct, train_size
            )
            validation_metrics = self.get_validation_metrics(
                validation_loss, validation_correct, val_size
            )

            print(f"\nEpoch {epoch+1} stats:")
            print(f"Train loss: {training_metrics.get_metric(MetricType.AVERAGE_LOSS):.4f}")
            print(f"Train acc:  {training_metrics.get_metric(MetricType.ACCURACY):.4f}")
            print(f"Val loss:   {validation_metrics.get_metric(MetricType.AVERAGE_LOSS):.4f}")
            print(f"Val acc:    {validation_metrics.get_metric(MetricType.ACCURACY):.4f}")

            results.training.append(training_metrics)
            results.validation.append(validation_metrics)

            # Store epoch-level debug stats
            debug_stats["train_output_dist"].append(np.concatenate(batch_outputs))
            debug_stats["val_output_dist"].append(np.concatenate(val_outputs))

        self.save_debug_stats(debug_stats)

        return results

    def get_training_metrics(
        self, absolute_loss: float, correct: int, data_size: float
    ) -> EpochMetrics:
        training_metrics = EpochMetrics()
        # training_metrics.add_metric(MetricType.ABSOLUTE_LOSS, absolute_loss)
        training_metrics.add_metric(MetricType.AVERAGE_LOSS, absolute_loss / data_size)
        training_metrics.add_metric(MetricType.ACCURACY, correct / data_size)
        return training_metrics

    def get_validation_metrics(
        self, val_loss: float, correct: int, data_size: float
    ) -> EpochMetrics:
        validation_metrics = EpochMetrics()
        # validation_metrics.add_metric(MetricType.ABSOLUTE_LOSS, val_loss)
        validation_metrics.add_metric(MetricType.AVERAGE_LOSS, val_loss / data_size)
        validation_metrics.add_metric(MetricType.ACCURACY, correct / data_size)
        return validation_metrics

    def save_debug_stats(self, debug_stats: dict, output_dir: str = "debug_outputs"):
        """
        saves debug statistics to a json file with timestamp

        like frieren storing her memories in that crystal thing fr fr
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # summarize the distributions to keep file size reasonable
        processed_stats = {
            "summary": {
                "timestamp": timestamp,
                "total_epochs": len(debug_stats["train_output_dist"]),
                "gradient_stats": {
                    "mean": np.mean(debug_stats["gradient_norms"]),
                    "std": np.std(debug_stats["gradient_norms"]),
                    "max": np.max(debug_stats["gradient_norms"]),
                    "min": np.min(debug_stats["gradient_norms"]),
                },
            },
            "epochs": [],
        }

        # process each epoch
        for epoch_idx in range(len(debug_stats["train_output_dist"])):
            epoch_data = {
                "epoch": epoch_idx + 1,
                "training": {
                    "output_distribution": {
                        "mean": float(np.mean(debug_stats["train_output_dist"][epoch_idx])),
                        "std": float(np.std(debug_stats["train_output_dist"][epoch_idx])),
                        "max": float(np.max(debug_stats["train_output_dist"][epoch_idx])),
                        "min": float(np.min(debug_stats["train_output_dist"][epoch_idx])),
                        "histogram": np.histogram(
                            debug_stats["train_output_dist"][epoch_idx], bins=20
                        )[0].tolist(),
                    }
                },
                "validation": {
                    "output_distribution": {
                        "mean": float(np.mean(debug_stats["val_output_dist"][epoch_idx])),
                        "std": float(np.std(debug_stats["val_output_dist"][epoch_idx])),
                        "max": float(np.max(debug_stats["val_output_dist"][epoch_idx])),
                        "min": float(np.min(debug_stats["val_output_dist"][epoch_idx])),
                        "histogram": np.histogram(
                            debug_stats["val_output_dist"][epoch_idx], bins=20
                        )[0].tolist(),
                    }
                },
                "gradients": {
                    # get gradients corresponding to this epoch
                    "mean": float(
                        np.mean(
                            debug_stats["gradient_norms"][
                                epoch_idx :: len(debug_stats["train_output_dist"])
                            ]
                        )
                    ),
                    "std": float(
                        np.std(
                            debug_stats["gradient_norms"][
                                epoch_idx :: len(debug_stats["train_output_dist"])
                            ]
                        )
                    ),
                    "max": float(
                        np.max(
                            debug_stats["gradient_norms"][
                                epoch_idx :: len(debug_stats["train_output_dist"])
                            ]
                        )
                    ),
                    "min": float(
                        np.min(
                            debug_stats["gradient_norms"][
                                epoch_idx :: len(debug_stats["train_output_dist"])
                            ]
                        )
                    ),
                },
            }
            processed_stats["epochs"].append(epoch_data)

        output_file = output_path / f"debug_stats_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(processed_stats, f, cls=NumpyEncoder, indent=2)

        print(f"Debug stats saved to {output_file}")
        return output_file

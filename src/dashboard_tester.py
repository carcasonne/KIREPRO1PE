from time import sleep

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score
from torch.utils.data import DataLoader, Dataset

from config import AudioClassifierConfig
from Core.Metrics import EpochMetrics, MetricType, Results
from Core.ResultsDashboardGenerator import ColorScheme, TrainingReporter


class DummyAudioDataset(Dataset):
    def __init__(self, size: int, img_height: int, img_width: int, channels: int):
        self.size = size
        self.shape = (channels, img_height, img_width)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(self.shape), torch.randint(0, 2, (1,))[0]


def simulate_predictions(n_samples: int, accuracy: float) -> tuple[np.ndarray, np.ndarray]:
    """simulate predictions with given accuracy"""
    true_labels = np.random.randint(0, 2, n_samples)
    predictions = np.where(np.random.random(n_samples) < accuracy, true_labels, 1 - true_labels)
    return true_labels, predictions


def generate_metrics_for_epoch(
    base_acc: float, base_loss: float, n_samples: int, noise_level: float = 0.05
) -> EpochMetrics:
    """generate all metrics for one epoch"""
    metrics = EpochMetrics()

    # simulate predictions
    y_true, y_pred = simulate_predictions(n_samples, base_acc)

    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # add all metrics
    metrics.add_metric(MetricType.AVERAGE_LOSS, base_loss + np.random.normal(0, noise_level))
    metrics.add_metric(MetricType.ACCURACY, (tp + tn) / n_samples)
    metrics.add_metric(MetricType.PRECISION, precision_score(y_true, y_pred))
    metrics.add_metric(MetricType.F1_SCORE, f1_score(y_true, y_pred))
    metrics.add_metric(MetricType.TRUE_POSITIVES, int(tp))
    metrics.add_metric(MetricType.TRUE_NEGATIVES, int(tn))
    metrics.add_metric(MetricType.FALSE_POSITIVES, int(fp))
    metrics.add_metric(MetricType.FALSE_NEGATIVES, int(fn))

    return metrics


def main():
    config = AudioClassifierConfig(
        # core params
        learning_rate=1e-3,
        batch_size=16,
        shuffle_batches=True,
        epochs=10,
        k_folds=5,
        optimizer="Adadelta",
        torch_seed=42,
        # model arch
        model_name="CNN Classifier from Paper",
        hidden_dims=[256, 128, 64],
        activation="gelu",
        # regularization
        dropout=0.1,
        weight_decay=0,
        # data specs
        sample_rate=16000,
        img_height=625,
        img_width=469,
        channels=3,
        duration=1,
        # misc
        notes="Testing dashboard generation with simulated data including confusion matrices",
        data_path="../audio_files_samples",
        output_path="../output",
        run_cuda=True,
    )

    # create dummy dataloaders
    training_data = DataLoader(
        DummyAudioDataset(1000, config.img_height, config.img_width, config.channels),
        batch_size=config.batch_size,
        shuffle=config.shuffle_batches,
    )

    validation_data = DataLoader(
        DummyAudioDataset(200, config.img_height, config.img_width, config.channels),
        batch_size=config.batch_size,
        shuffle=False,
    )

    testing_data = DataLoader(
        DummyAudioDataset(100, config.img_height, config.img_width, config.channels),
        batch_size=config.batch_size,
        shuffle=False,
    )

    # simulate k-fold results
    k_fold_results = []
    n_epochs = 10

    for k in range(config.k_folds):
        results = Results()

        # generate somewhat realistic learning curves
        base_loss = np.exp(np.linspace(0, -2, n_epochs))  # exponential decay for loss
        base_acc = 0.5 + 0.4 * (1 - np.exp(-np.linspace(0, 3, n_epochs)))  # accuracy curve

        # add metrics for each epoch
        for i in range(n_epochs):
            # training metrics (slightly different for each fold)
            results.training.append(
                generate_metrics_for_epoch(
                    base_acc=base_acc[i] * (1 + 0.1 * np.random.random()),
                    base_loss=base_loss[i] * (1 + 0.1 * np.random.random()),
                    n_samples=1000,
                    noise_level=0.05,
                )
            )

            # validation metrics with slightly worse performance
            results.validation.append(
                generate_metrics_for_epoch(
                    base_acc=0.95 * base_acc[i] * (1 + 0.1 * np.random.random()),
                    base_loss=1.2 * base_loss[i] * (1 + 0.1 * np.random.random()),
                    n_samples=200,
                    noise_level=0.07,
                )
            )

            # testing metrics (just for final epoch)
            if i == n_epochs - 1:
                results.testing.append(
                    generate_metrics_for_epoch(
                        base_acc=0.97 * base_acc[i],
                        base_loss=1.1 * base_loss[i],
                        n_samples=100,
                        noise_level=0.06,
                    )
                )

        k_fold_results.append(results)

    # generate report for each color scheme
    color_schemes = [ColorScheme.FRIEREN]
    for color_scheme in color_schemes:
        print(f"\nGenerating report with {color_scheme.name} theme...")
        reporter = TrainingReporter(
            config=config,
            training_data=training_data,
            validation_data=validation_data,
            testing_data=testing_data,
            color_scheme=color_scheme,
            base_dir="../training_runs",
        )

        # generate k-fold report
        report_path = reporter.generate_report_k_folds(
            k_fold_results=k_fold_results,
            complete_time=10.0,  # dummy training time
        )
        print(f"Report generated at: {report_path}")

        # small delay to avoid file conflicts
        sleep(1)


if __name__ == "__main__":
    main()

from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import *
from config import AudioClassifierConfig
from Core.Metrics import EpochMetrics, MetricType, Results
from Core.ResultsDashboardGenerator import ColorScheme, TrainingReporter


# dummy dataset class that just returns tensors of the right shape
class DummyAudioDataset(Dataset):
    def __init__(self, size: int, img_height: int, img_width: int, channels: int):
        self.size = size
        self.shape = (channels, img_height, img_width)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # return a spectogram-shaped tensor and a random label
        return torch.randn(self.shape), torch.randint(0, 2, (1,))[0]


config = AudioClassifierConfig(
    # core params
    learning_rate=1e-3,
    batch_size=16,
    shuffle_batches=False,
    epochs=2,
    optimizer="Adadelta",
    torch_seed=None,
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
    notes="Testing dashboard generation with simulated data",
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

# simulate training results with some realistic patterns
results = Results()

# number of epochs to simulate
n_epochs = 20

# generate somewhat realistic learning curves
base_loss = np.exp(np.linspace(0, -2, n_epochs))  # exponential decay for loss
train_noise = np.random.normal(0, 0.05, n_epochs)  # add some noise
val_noise = np.random.normal(0, 0.07, n_epochs)  # slightly more noise for validation

base_acc = 0.5 + 0.4 * (1 - np.exp(-np.linspace(0, 3, n_epochs)))  # accuracy curve
acc_train_noise = np.random.normal(0, 0.02, n_epochs)
acc_val_noise = np.random.normal(0, 0.03, n_epochs)

# add metrics for each epoch
for i in range(n_epochs):
    # training metrics
    train_metrics = EpochMetrics()
    train_metrics.add_metric(MetricType.ABSOLUTE_LOSS, float(base_loss[i] + train_noise[i]))
    train_metrics.add_metric(MetricType.ACCURACY, float(base_acc[i] + acc_train_noise[i]))
    train_metrics.add_metric(
        MetricType.AVERAGE_LOSS, float(base_loss[i] + train_noise[i]) / (i + 1)
    )
    results.training.append(train_metrics)

    # validation metrics with slightly worse performance
    val_metrics = EpochMetrics()
    val_metrics.add_metric(MetricType.ABSOLUTE_LOSS, float(1.2 * base_loss[i] + val_noise[i]))
    val_metrics.add_metric(MetricType.ACCURACY, float(0.95 * base_acc[i] + acc_val_noise[i]))
    val_metrics.add_metric(
        MetricType.AVERAGE_LOSS, float(1.2 * base_loss[i] + val_noise[i]) / (i + 1)
    )
    results.validation.append(val_metrics)

    # testing metrics (just add to final epoch)
    if i == n_epochs - 1:
        test_metrics = EpochMetrics()
        test_metrics.add_metric(
            MetricType.ABSOLUTE_LOSS, float(1.1 * base_loss[i] + np.random.normal(0, 0.05))
        )
        test_metrics.add_metric(
            MetricType.ACCURACY, float(0.97 * base_acc[i] + np.random.normal(0, 0.02))
        )
        test_metrics.add_metric(
            MetricType.AVERAGE_LOSS,
            float(1.1 * base_loss[i] + np.random.normal(0, 0.05)) / n_epochs,
        )
        results.testing.append(test_metrics)

# generate report and do it for all the color themes
for color_scheme in ColorScheme:
    reporter = TrainingReporter(
        config=config,
        training_data=training_data,
        validation_data=validation_data,
        testing_data=testing_data,
        color_scheme=color_scheme,
        base_dir="../training_runs",
    )
    report_path = reporter.generate_report(results)
    print(f"Report generated at: {report_path}")
    sleep(1)

from pathlib import Path
from typing import List, Optional, Tuple, Union

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from Core.DataType import DataType
from Core.Metrics import MetricType, Results

matplotlib.use("TkAgg")


class PlotVisualizer:
    @staticmethod
    def as_spectrogram(data: np.ndarray, figsize: Tuple[int, int]) -> None:
        """print the nd array as a spectogram"""
        plt.figure(figsize=figsize)
        librosa.display.specshow(data, x_axis="time", y_axis="log", cmap="viridis")
        plt.colorbar(format="%+2.0f dB")
        plt.title("STFT Spectrogram (dB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.show()

    @staticmethod
    def as_heatmap(
        data: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: str = "Heatmap",
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """generic heatmap for when u just wanna see the numbers"""
        plt.figure(figsize=figsize)
        plt.imshow(data, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title(title)
        plt.show()


class MetricsVisualizer:
    @staticmethod
    def plot_metric(
        results: Results,
        metric_type: MetricType,
        data_types: Optional[List[DataType]] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
    ) -> None:
        """fr fr this plots your metrics across epochs frfr no cap"""

        if data_types is None:
            data_types = [DataType.TRAINING, DataType.VALIDATION, DataType.TESTING]

        plt.figure(figsize=figsize)

        for data_type in data_types:
            data = getattr(results, data_type.value)
            if not data:  # skip if empty (frieren would approve of this defensive programming)
                continue

            epochs = range(1, len(data) + 1)
            values = [epoch.get_metric(metric_type) for epoch in data]

            plt.plot(epochs, values, label=data_type.value.capitalize(), marker="o")

        plt.xlabel("Epoch")
        plt.ylabel(metric_type.name.replace("_", " ").title())
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        if title:
            plt.title(title)
        else:
            plt.title(f'{metric_type.name.replace("_", " ").title()} vs Epoch')

        plt.show()

    @staticmethod
    def plot_all_metrics(
        results: Results,
        data_types: Optional[List[DataType]] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> None:
        """deadass plots ALL your metrics at once"""

        if data_types is None:
            data_types = [DataType.TRAINING, DataType.VALIDATION, DataType.TESTING]

        metrics = list(MetricType)
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for data_type in data_types:
                data = getattr(results, data_type.value)
                if not data:
                    continue

                epochs = range(1, len(data) + 1)
                values = [epoch.get_metric(metric) for epoch in data]

                ax.plot(epochs, values, label=data_type.value.capitalize(), marker="o")

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.name.replace("_", " ").title())
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend()
            ax.set_title(f'{metric.name.replace("_", " ").title()}')

        plt.tight_layout()
        plt.show()

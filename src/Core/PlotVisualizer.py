from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


class PlotVisualizer:
    @staticmethod
    def as_spectrogram(data: np.ndarray, figsize: Tuple[int, int]) -> None:
        """print the nd array as a spectogram"""
        # plt.figure(figsize=figsize)
        # librosa.display.specshow(data, x_axis="time", y_axis="log", cmap="viridis")
        # plt.colorbar(format="%+2.0f dB")
        # plt.title(title)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Frequency (Hz)")
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

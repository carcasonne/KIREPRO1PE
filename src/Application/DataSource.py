import os
from enum import Enum
from io import BytesIO

import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from Core.DataType import DataType


class AudioLabel(Enum):
    FAKE = 0
    REAL = 1


class AudioData(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform,
        sample_rate: int,
        subset: DataType,
        audio_duration_seconds: int,
    ):
        self.transform = transform
        self.sample_rate = sample_rate
        self.subset = subset
        self.audio_duration_seconds = audio_duration_seconds
        self.files = []

        for label in AudioLabel:
            class_path = os.path.join(root_dir, subset.value, label.name.lower())
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                self.files.append((file_path, label.value))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        spec = self.convert_to_spectrogram(file_path)
        img = self.spectrogram_to_rgb(spec)
        # shit doesnt normalize yet
        # if self.transform:
        #     spec = self.transform(spec)
        return img, label

    def convert_to_spectrogram(self, filepath):
        audio, _ = librosa.load(filepath, sr=self.sample_rate, duration=self.audio_duration_seconds)
        spec = librosa.stft(audio)
        return librosa.amplitude_to_db(np.abs(spec), ref=np.max)

    # Converts the spectrogram to an image
    # TODO: Clean this up a little.
    # TODO: Also have to consider the fact that using librosa to display the spectrogram
    # TODO: gives a different 'image' than when using matplotlib, write in report at least
    def spectrogram_to_rgb(self, spectrogram):
        fig, ax = plt.subplots(figsize=(10, 6))
        librosa.display.specshow(
            spectrogram, sr=self.sample_rate, x_axis="time", y_axis="log", cmap="viridis"
        )
        plt.axis("off")

        # Save the plot to a BytesIO object
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Load the image from the buffer and convert to an RGB array
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        rgb_array = np.array(image)
        buf.close()

        # Have to convert from uint8 to float32 and then normalize
        rgb_image_array_float = rgb_array.astype(np.float32) / 255.0
        # Swaps the dimensions from 462, 775, 3 to 3, 462, 775
        rgb_image_array_transposed = np.transpose(rgb_image_array_float, (2, 0, 1))
        return rgb_image_array_transposed


class LocalDataSource:
    def __init__(self, root_dir: str, sample_rate: int, audio_duration_seconds: int, transform):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.audio_duration_seconds = audio_duration_seconds
        self.transform = transform

    def get_data_loader(self, subset: DataType, batch_size: int, shuffle: bool) -> AudioData:
        data = AudioData(
            root_dir=self.root_dir,
            transform=self.transform,
            sample_rate=self.sample_rate,
            subset=subset,
            audio_duration_seconds=self.audio_duration_seconds,
        )
        return DataLoader(data, batch_size, shuffle)

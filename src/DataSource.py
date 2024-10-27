import os
from enum import Enum

import librosa
import numpy as np
from torch.utils.data import DataLoader, Dataset


class DataType(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"

class AudioLabel(Enum):
    FAKE = 0
    REAL = 1

class LocalDataSource():
    def __init__(self,
                 root_dir : str,
                 sample_rate: int,
                 audio_duration_seconds: int,
                 transform):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.audio_duration_seconds = audio_duration_seconds
        self.transform = transform

    def get_data_loader(self, subset: DataType, batch_size: int, shuffle: bool):
        data = AudioData(root_dir=self.root_dir,
                         transform=self.transform,
                         sample_rate=self.sample_rate,
                         subset=subset,
                         audio_duration_seconds=self.audio_duration_seconds)
        return DataLoader(data, batch_size, shuffle)

class AudioData(Dataset):
    def __init__(self,
                 root_dir: str,
                 transform,
                 sample_rate: int,
                 subset: DataType,
                 audio_duration_seconds: int,):
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
        # shit doesnt normalize yet
        # if self.transform:
        #     spec = self.transform(spec)
        return spec, label

    def convert_to_spectrogram(self, filepath):
        audio, _ = librosa.load(
            filepath, sr=self.sample_rate, duration=self.audio_duration_seconds)
        spec = librosa.stft(audio)
        return librosa.amplitude_to_db(np.abs(spec), ref=np.max)


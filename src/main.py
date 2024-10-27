from torchvision import transforms

from Application.ClonedAudioDetector import CNNClassifier
from Application.DataProcessor import DataProcessor
from Application.DataSource import DataType, LocalDataSource
from config import *
from Core.PlotVisualizer import PlotVisualizer

transform_normalization = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)

# Load data
data_source = LocalDataSource(DATA_PATH, SAMPLE_RATE, DURATION, transform_normalization)
training_data = data_source.get_data_loader(DataType.TRAINING, BATCH_SIZE, SHUFFLE)
validation_data = data_source.get_data_loader(DataType.TRAINING, BATCH_SIZE, SHUFFLE)
testing_data = data_source.get_data_loader(DataType.TRAINING, BATCH_SIZE, SHUFFLE)

# Define our classifier network
classifier = CNNClassifier(no_channels=3)

# Define data processing pipeline
data_processor = DataProcessor(classifier, True)

# just test for now

spec = training_data.get_sample()
PlotVisualizer.as_spectrogram(data=spec.numpy(), figsize=(10, 6))

from torchvision import transforms

from Application.ClonedAudioDetector import CNNClassifier
from Application.DataProcessor import DataProcessor
from Application.DataSource import DataType, LocalDataSource
from config import *

transform_normalization = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)

# Load data
data_source = LocalDataSource(DATA_PATH, SAMPLE_RATE, DURATION, transform_normalization)
training_data = data_source.get_data_loader(DataType.TRAINING, BATCH_SIZE, SHUFFLE)
validation_data = data_source.get_data_loader(DataType.VALIDATION, BATCH_SIZE, SHUFFLE)
testing_data = data_source.get_data_loader(DataType.TESTING, BATCH_SIZE, SHUFFLE)

# Define our classifier network
classifier = CNNClassifier(no_channels=1)

# Define data processing pipeline
data_processor = DataProcessor(classifier, True)

# just test for now

# from Core.Visualizers import PlotVisualizer
# spec = training_data.get_sample()
# PlotVisualizer.as_spectrogram(data=spec.numpy(), figsize=(10, 6))

# try to run this bitch
# ok i cant run this bitch it crashes:
# RuntimeError: Given groups=1, weight of size [32, 1, 2, 2], expected input[1, 32, 1025, 32] to have 1 channels, but got 32 channels instead
print("Batch size: " + str(BATCH_SIZE))
results = data_processor.process(training_data, validation_data, 1)

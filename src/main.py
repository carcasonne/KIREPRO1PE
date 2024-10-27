from torchvision import transforms

from ClonedAudioDetector import CNNClassifier
from config import *
from DataProcessor import DataProcessor
from DataSource import DataType, LocalDataSource

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

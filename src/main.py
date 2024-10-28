from torchvision import transforms

from Application.ClonedAudioDetector import CNNClassifier
from Application.DataProcessor import DataProcessor
from Application.DataSource import DataType, LocalDataSource
from config import *
from config import AudioClassifierConfig
from Core.ResultsDashboardGenerator import TrainingReporter

config = AudioClassifierConfig(
    # core params
    learning_rate=1e-3,
    batch_size=32,
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
    # data specs (some of these things should probs be under model architecture tbh)
    sample_rate=16000,
    img_height=625,
    img_width=469,
    channels=3,
    duration=1,
    # misc
    notes="testing hypothesis about attention patterns",
    data_path="../audio_files_samples",
    output_path="../output",
    run_coda=True,
)

transform_normalization = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)

# Load data
data_source = LocalDataSource(
    config.data_path, config.sample_rate, config.duration, transform_normalization
)
training_data = data_source.get_data_loader(
    DataType.TRAINING, config.batch_size, config.shuffle_batches
)
validation_data = data_source.get_data_loader(
    DataType.VALIDATION, config.batch_size, config.shuffle_batches
)
testing_data = data_source.get_data_loader(
    DataType.TESTING, config.batch_size, config.shuffle_batches
)

# Define our classifier network
classifier = CNNClassifier(no_channels=config.channels)

# Define data pipeline
data_processor = DataProcessor(classifier, config.run_coda)
reporter = TrainingReporter(config=config, base_dir="../training_runs")

# Process data
test = data_processor.process(training_data, validation_data, config.epochs)

# Interpret data
report_path = reporter.generate_report(results)
print(f"Report generated at: {report_path}")

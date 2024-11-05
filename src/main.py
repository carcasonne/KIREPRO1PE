from torchvision import transforms

from Application.ClonedAudioDetector import CNNClassifier
from Application.DataProcessor import DataProcessor
from Application.DataSource import DataType, LocalDataSource
from config import *
from config import AudioClassifierConfig
from Core.ResultsDashboardGenerator import ColorScheme, TrainingReporter

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
    # data specs (some of these things should probs be under model architecture tbh)
    sample_rate=16000,
    img_height=625,
    img_width=469,
    channels=3,
    duration=1,
    # misc
    notes="The data was trained on the Cloned Audio CNN Classifier defined in the paper: 'Fighting AI with AI: Fake Speech Detection using Deep Learning' by Malik & Changalvala.",
    data_path="../audio_files_samples",
    output_path="../output",
    run_cuda=True,
)

transform_normalization = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
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
data_processor = DataProcessor(classifier, config.run_cuda)
reporter = TrainingReporter(
    config=config,
    training_data=training_data,
    validation_data=validation_data,
    testing_data=testing_data,
    color_scheme=ColorScheme.FRIEREN,
    base_dir="../training_runs",
)

# Process data
results = data_processor.process(
    training_data, validation_data, config.epochs, config.learning_rate
)

# Interpret data
report_path = reporter.generate_report(results)
print(f"Report generated at: {report_path}")

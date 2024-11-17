from torchvision import transforms

from Application.ClonedAudioDetector import CNNClassifier
from Application.DataProcessor import DataProcessor
from Application.DataSource import DataType, LocalDataSource
from config import *
from config import AudioClassifierConfig
from Core.ResultsDashboardGenerator import ColorScheme, TrainingReporter

config = AudioClassifierConfig(
    # core params
    learning_rate=None,
    batch_size=16,
    shuffle_batches=True,
    epochs=10,
    k_folds=5,  # if k-folds = None, do not use k-fold method
    optimizer="Adadelta",
    torch_seed=None,
    # model arch
    model_name="CNN Classifier from Paper",
    hidden_dims=[32],
    activation="gelu",
    # regularization
    dropout=0.1,
    weight_decay=0,
    # data specs (some of these things should probs be under model architecture tbh)
    sample_rate=16000,
    img_height=625,
    img_width=469,
    channels=3,
    duration=2,
    # misc
    notes="The data was trained on the Cloned Audio CNN Classifier defined in the paper: 'Fighting AI with AI: Fake Speech Detection using Deep Learning' by Malik & Changalvala.",
    data_path="../audio_files_fake_from_paper",
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

k_fold_data = data_source.get_k_fold_dataset()

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

results, complete_time = data_processor.process_k_fold(
    CNNClassifier, k_fold_data, config.k_folds, config.epochs, config.batch_size, config.run_cuda
)

# Interpret data
report_path = reporter.generate_report_k_folds(k_fold_results=results, complete_time=complete_time)
print(f"Report generated at: {report_path}")

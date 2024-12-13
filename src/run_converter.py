from Application.ClonedAudioDetector import CNNClassifier
from Application.DataProcessor import DataProcessor
from config import *
from config import AudioClassifierConfig
from Core.ResultsDashboardGenerator import ColorScheme, TrainingReporter
from Core.ResultsLoader import ResultsLoader

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
    data_path="../Ar-DAD_Arabic_Diversified",
    output_path="../output",
    run_cuda=True,
)

original_result_name = "20241119_083246"
original_result_path = f"training_runs/{original_result_name}/kfold_data.json"
color_scheme = ColorScheme.LIGHT

# Define our classifier network
classifier = CNNClassifier(no_channels=config.channels)

# Define data pipeline
data_processor = DataProcessor(classifier, config.run_cuda)
reporter = TrainingReporter(
    config=config,
    training_data=None,
    validation_data=None,
    testing_data=None,
    color_scheme=color_scheme,
    base_dir="./converted_runs",
)

results = ResultsLoader.load(original_result_path)
report_path = reporter.generate_report_k_folds(k_fold_results=results, complete_time=0)
print(f"Report generated at: {report_path}")

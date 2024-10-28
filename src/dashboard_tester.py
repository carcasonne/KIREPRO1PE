from config import AudioClassifierConfig
from Core.Metrics import EpochMetrics, MetricType, Results
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

# simulate some results
results = Results()

# add some training data
for i in range(10):
    train_metrics = EpochMetrics()
    train_metrics.add_metric(MetricType.ABSOLUTE_LOSS, 1.0 - i * 0.1)
    train_metrics.add_metric(MetricType.ACCURACY, 0.7 + i * 0.02)
    results.training.append(train_metrics)

    val_metrics = EpochMetrics()
    val_metrics.add_metric(MetricType.ABSOLUTE_LOSS, 1.2 - i * 0.08)
    val_metrics.add_metric(MetricType.ACCURACY, 0.65 + i * 0.02)
    results.validation.append(val_metrics)

# generate report
reporter = TrainingReporter(config=config, base_dir="../training_runs")
report_path = reporter.generate_report(results)
print(f"Report generated at: {report_path}")

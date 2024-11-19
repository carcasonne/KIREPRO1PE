import colorsys
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

from config import AudioClassifierConfig
from Core.Metrics import MetricType, Results


# just messing around, easy to add new stuff
class ColorScheme(Enum):
    LIGHT = auto()
    DARK = auto()
    SYNTHWAVE = auto()
    KANAGAWA = auto()
    CATPPUCCIN_MOCHA = auto()
    CATPPUCCIN_MACCHIATO = auto()
    IBM_CARBON = auto()
    FRIEREN = auto()
    DRACULA = auto()
    GRUVBOX = auto()
    NORD = auto()

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ").title()


# Bit redundant, but makes it obvious which parts of the data loader the report is going to display
@dataclass
class DatasetInfo:
    training_size: int
    validation_size: int
    testing_size: int
    batch_size: int
    shuffle: bool
    sample_rate: int
    duration: int
    img_dims: tuple[int, int]
    channels: int
    k_folds: int


class MetricPlotter:
    PHASE_COLORS = {
        "training": "#2ecc71",
        "validation": "#3498db",
        "testing": "#e74c3c",
    }

    CONFUSION_METRICS = {
        MetricType.TRUE_POSITIVES,
        MetricType.TRUE_NEGATIVES,
        MetricType.FALSE_POSITIVES,
        MetricType.FALSE_NEGATIVES,
    }

    @staticmethod
    def extract_metric_data(results: Results, metric_type: MetricType) -> Dict[str, List[float]]:
        # special handling for confusion matrix metrics which are integers
        convert_fn = int if metric_type in MetricPlotter.CONFUSION_METRICS else float

        return {
            "training": [convert_fn(epoch.get_metric(metric_type)) for epoch in results.training],
            "validation": [
                convert_fn(epoch.get_metric(metric_type)) for epoch in results.validation
            ],
            "testing": [convert_fn(epoch.get_metric(metric_type)) for epoch in results.testing],
        }

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, title: str, theme: dict) -> plt.Figure:
        """plots a single confusion matrix using theme colors"""
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # create a custom colormap from theme colors
        colors = [theme["bg-secondary"], theme["accent-subtle"], theme["accent"]]

        # convert hex to rgb for colormap
        rgb_colors = [
            tuple(int(h.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)) for h in colors
        ]

        custom_cmap = LinearSegmentedColormap.from_list("custom", rgb_colors, N=100)

        # normalize data for better color distribution
        norm = plt.Normalize(vmin=0, vmax=np.max(cm))

        # plot matrix with custom colormap but don't create colorbar yet
        im = ax.imshow(cm, interpolation="nearest", cmap=custom_cmap, norm=norm)

        # create a single colorbar with proper styling
        cbar = fig.colorbar(im, ax=ax)
        cbar.outline.set_edgecolor(theme["border"])
        cbar.ax.yaxis.set_tick_params(color=theme["text-secondary"])
        cbar.ax.yaxis.set_ticks_position("right")  # keep it on the right
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=theme["text-secondary"])

        # add value labels to each cell with adaptive color
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] / np.max(cm) > 0.5 else theme["text-primary"]
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=12,
                    fontweight="bold",
                )

        ax.set_title(title)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        # add labels
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Negative", "Positive"])

        # apply theme
        ax.set_facecolor(theme["bg-secondary"])
        fig.set_facecolor(theme["bg-primary"])
        ax.set_title(ax.get_title(), color=theme["text-primary"])
        ax.set_xlabel(ax.get_xlabel(), color=theme["text-secondary"])
        ax.set_ylabel(ax.get_ylabel(), color=theme["text-secondary"])
        ax.tick_params(colors=theme["text-secondary"])

        return fig

    @staticmethod
    def get_confusion_matrix_for_phase(results: Results, phase: str) -> np.ndarray:
        """extracts confusion matrix from final epoch of given phase"""
        phase_data = getattr(results, phase)[-1]  # get last epoch
        return np.array(
            [
                [
                    phase_data.get_metric(MetricType.TRUE_NEGATIVES),
                    phase_data.get_metric(MetricType.FALSE_POSITIVES),
                ],
                [
                    phase_data.get_metric(MetricType.FALSE_NEGATIVES),
                    phase_data.get_metric(MetricType.TRUE_POSITIVES),
                ],
            ]
        )

    @staticmethod
    def get_average_confusion_matrix(results_list: List[Results], phase: str) -> np.ndarray:
        """computes average confusion matrix across all folds for final epoch"""
        matrices = []
        for results in results_list:
            matrices.append(MetricPlotter.get_confusion_matrix_for_phase(results, phase))
        return np.mean(matrices, axis=0).astype(int)  # average and round to integers

    @staticmethod
    def should_plot_separately(metric_type: MetricType) -> bool:
        """determine if metric should be plotted in main graphs"""
        return metric_type not in MetricPlotter.CONFUSION_METRICS


class TrainingReporter:
    def __init__(
        self,
        config: AudioClassifierConfig,
        training_data: DataLoader,
        validation_data: DataLoader,
        testing_data: DataLoader,
        color_scheme: ColorScheme = ColorScheme.LIGHT,
        base_dir: str = "training_runs",
    ):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, self.timestamp)
        self.plot_dir = os.path.join(self.run_dir, "plots")
        self.config = config
        self.color_scheme = color_scheme
        self.metrictypes = []

        self.system_info = {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "torch_version": torch.__version__,
            "ram": f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }

        self.dataset_info = DatasetInfo(
            training_size=len(training_data.dataset),
            validation_size=len(validation_data.dataset) if validation_data else 0,
            testing_size=len(testing_data.dataset) if testing_data else 0,
            batch_size=config.batch_size,
            shuffle=config.shuffle_batches,
            sample_rate=config.sample_rate,
            duration=config.duration,
            img_dims=(config.img_height, config.img_width),
            channels=config.channels,
            k_folds=config.k_folds,
        )

        os.makedirs(self.plot_dir, exist_ok=True)

    def _generate_plots(self, results: Results) -> Dict[MetricType, str]:
        plots_info = {"metrics": {}, "confusion": []}

        # standard metric plots
        for metric_type in self.metrictypes:
            if not MetricPlotter.should_plot_separately(metric_type):
                continue

            metric_data = MetricPlotter.extract_metric_data(results, metric_type)
            if not any(len(data) > 0 for data in metric_data.values()):
                continue

            plt.figure(figsize=(10, 6))
            for phase, data in metric_data.items():
                if len(data) > 0:
                    plt.plot(
                        data,
                        label=phase.capitalize(),
                        color=MetricPlotter.PHASE_COLORS[phase],
                        alpha=0.8,
                    )

            self.style_plot(f'{metric_type.name.replace("_", " ").title()} Over Epochs')

            plot_name = f"{metric_type.name.lower()}.png"
            plot_path = os.path.join(self.plot_dir, plot_name)
            plt.savefig(plot_path, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
            plt.close()

            plots_info["metrics"][metric_type] = os.path.relpath(plot_path, self.run_dir)

        # confusion matrix plots for first and last epoch
        theme = self._get_theme_css()[self.color_scheme]
        for phase in ["training", "validation"]:
            phase_data = getattr(results, phase)
            if phase_data:
                # plot first and last epoch
                for epoch_idx in [0, -1]:
                    cm_path = MetricPlotter.plot_confusion_matrix(
                        results, epoch_idx, phase, self.plot_dir, theme
                    )
                    plots_info["confusion"].append(cm_path)

        return plots_info

    def generate_report(self, results: Results, complete_time: float) -> str:
        plots_info = self._generate_plots(results)
        self.metrictypes = results.get_present_metrics()
        self.save_run_data(results)
        return self._generate_dashboard(
            plots_info=plots_info, results=results, complete_time=complete_time
        )

    def generate_report_k_folds(self, k_fold_results: List[Results], complete_time: float) -> str:
        self._save_kfold_data(k_fold_results)
        self.metrictypes = Results.get_present_metrics_from_list(k_fold_results)
        return self._generate_dashboard(
            plots_info=None,
            results=None,
            complete_time=complete_time,
            k_fold_results=k_fold_results,
        )

    def _generate_dashboard(
        self,
        plots_info: Dict[MetricType, str],
        results: Results,
        complete_time: float,
        k_fold_results: List[Results] | None = None,
    ) -> str:
        k = len(k_fold_results) if k_fold_results else None
        k_fold_plots = None
        k_fold_stats = None
        graphs_metrics_section = None
        confusion_section = None

        if k_fold_results:
            k_fold_plots = self._generate_kfold_plots(k_fold_results, k, self.plot_dir)
            k_fold_stats = self._compute_kfold_statistics(k_fold_results)
            graphs_metrics_section = self._get_k_folds_section_html(k_fold_plots, k_fold_stats)
            confusion_plots = self._generate_kfold_confusion_matrices(k_fold_results)
            confusion_section = self._get_confusion_section_html(confusion_plots, is_kfold=True)
        else:
            graphs_metrics_section = self._get_plots_section_html(
                plots_info
            ) + self._get_metrics_section_html(results)
            confusion_plots = self._generate_confusion_matrices(results)
            confusion_section = self._get_confusion_section_html(confusion_plots)

        dataloader_section = self._get_dataloader_section_html()
        run_config_section = self._get_config_section_html()

        html = f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Training Run {self.timestamp}</title>
                <style>
                    {self._get_css()}
                </style>
            </head>
            <body>
                <div class="card">
                    <h1>Training Run {self.timestamp}</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Color Scheme: {self.color_scheme.display_name}</p>
                    <p>Time to train model: {complete_time/60:.2f} minutes</p>
                </div>

                <div class="config-grid">
                    {dataloader_section}
                    {run_config_section}
                </div>
                {confusion_section}
                {graphs_metrics_section}
            </body>
        </html>
        """

        output_path = os.path.join(self.run_dir, "report.html")
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _get_dataloader_section_html(self) -> str:
        return f"""
            <div class="config-card">
                <h2>DataLoader Configuration</h2>
                <table>
                    <tr><th colspan="2">Dataset Information</th></tr>
                    <tr><td>Training Set Size</td><td>{self.dataset_info.training_size:,} samples</td></tr>
                    <tr><td>Validation Set Size</td><td>{self.dataset_info.validation_size:,} samples</td></tr>
                    <tr><td>Testing Set Size</td><td>{self.dataset_info.testing_size:,} samples</td></tr>
                    <tr><td>Total Dataset Size</td>
                        <td>{self.dataset_info.training_size + self.dataset_info.validation_size + self.dataset_info.testing_size:,} samples</td></tr>
                    
                    <tr><th colspan="2">Batch Processing</th></tr>
                    <tr><td>Batch Size</td><td>{self.dataset_info.batch_size}</td></tr>
                    <tr><td>Shuffle Enabled</td><td>{self.dataset_info.shuffle}</td></tr>
                    <tr><td>Training Batches</td><td>{self.dataset_info.training_size // self.dataset_info.batch_size}</td></tr>
                    <tr><td>K Folds</td><td>{self.dataset_info.k_folds}</td></tr>
                    
                    <tr><th colspan="2">Audio Processing</th></tr>
                    <tr><td>Sample Rate</td><td>{self.dataset_info.sample_rate} Hz</td></tr>
                    <tr><td>Duration</td><td>{self.dataset_info.duration} seconds</td></tr>
                    
                    <tr><th colspan="2">Spectogram Configuration</th></tr>
                    <tr><td>Image Dimensions</td><td>{self.dataset_info.img_dims[0]}x{self.dataset_info.img_dims[1]}</td></tr>
                    <tr><td>Channels</td><td>{self.dataset_info.channels}</td></tr>
                    <tr><td>Pixels per Sample</td>
                        <td>{self.dataset_info.img_dims[0] * self.dataset_info.img_dims[1] * self.dataset_info.channels:,}</td></tr>
                </table>
            </div>
        """

    def _get_config_section_html(self) -> str:
        return f"""
            <div class="config-card">
                <h2>Run Configuration</h2>
                <table>
                    <tr><th colspan="2">Training Parameters</th></tr>
                    <tr><td>Learning Rate</td><td>{self.config.learning_rate}</td></tr>
                    <tr><td>Optimizer</td><td>{self.config.optimizer}</td></tr>
                    <tr><td>Torch Seed</td><td>{self.config.torch_seed}</td></tr>
                    <tr><td>Epochs</td><td>{self.config.epochs}</td></tr>
                    
                    <tr><th colspan="2">Model Architecture</th></tr>
                    <tr><td>Model</td><td>{self.config.model_name}</td></tr>
                    <tr><td>Hidden Dims</td><td>{self.config.hidden_dims}</td></tr>
                    <tr><td>Activation</td><td>{self.config.activation}</td></tr>
                    
                    <tr><th colspan="2">Regularization</th></tr>
                    <tr><td>Dropout</td><td>{self.config.dropout}</td></tr>
                    <tr><td>Weight Decay</td><td>{self.config.weight_decay}</td></tr>

                    <tr><th colspan="2">System Info</th></tr>
                    {"".join(f"<tr><td>{k.replace('_', ' ').title()}</td><td>{v}</td></tr>" for k, v in self.system_info.items())}
                </table>
                
                {f'<div class="notes"><h3>Notes</h3><p>{self.config.notes}</p></div>' if self.config.notes else ''}
            </div>
        """

    def _get_plots_section_html(self, plots_info: Dict[str, Any]) -> str:
        html = '<div class="card"><h2>Training Metrics</h2>'

        # standard metrics
        html += '<div class="metric-grid">'
        for metric_type, plot_path in plots_info["metrics"].items():
            html += f"""
                <div>
                    <h3>{metric_type.name.replace("_", " ").title()}</h3>
                    <img src="{plot_path}" alt="{metric_type.name}">
                </div>
            """
        html += "</div>"

        # confusion matrices
        if plots_info["confusion"]:
            html += '<h2>Confusion Matrices</h2><div class="confusion-grid">'
            for plot_path in plots_info["confusion"]:
                html += f'<img src="{plot_path}" alt="Confusion Matrix">'
            html += "</div>"

        html += "</div>"
        return html

    def _get_metrics_section_html(self, results: Results) -> str:
        html = '<div class="card"><h2>Final Metrics</h2>'
        for phase in ["training", "validation", "testing"]:
            phase_data = getattr(results, phase)
            if phase_data:
                html += f"""
                    <h3 class="phase-header">{phase.capitalize()} Results</h3>
                    <table>
                        <tr><th>Metric</th><th>Final Value</th></tr>
                """
                final_metrics = phase_data[-1].metrics
                for metric_type, value in final_metrics.items():
                    html += f"""
                        <tr>
                            <td>{metric_type.name.replace("_", " ").title()}</td>
                            <td>{value:.4f}</td>
                        </tr>c
                    """
                html += "</table>"

        html += "</div>"
        return html

    def _get_k_folds_section_html(self, k_fold_plots, k_fold_stats) -> str:
        # stats table remains the same
        html = '<div class="card"><h2>Cross Validation Statistics</h2>'

        # Keep existing stats table generation...
        for phase in ["training", "validation", "testing"]:
            if k_fold_stats[phase]["metrics"]:
                html += f"<h3>{phase.capitalize()} Statistics</h3>"
                html += """
                    <div class="table-container">
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Mean</th>
                                <th>Std Dev</th>
                                <th>Min</th>
                                <th>Max</th>
                                <th>Convergence (epochs)</th>
                                <th>Stability</th>
                            </tr>
                """

                for metric_type in k_fold_stats[phase]["metrics"].keys():
                    metric_stats = k_fold_stats[phase]["metrics"][metric_type]
                    convergence = k_fold_stats[phase]["convergence"][metric_type]
                    stability = k_fold_stats[phase]["stability"][metric_type]

                    html += f"""
                        <tr>
                            <td>{metric_type.name.replace("_", " ").title()}</td>
                            <td>{metric_stats['mean']:.4f}</td>
                            <td>{metric_stats['std']:.4f}</td>
                            <td>{metric_stats['min']:.4f}</td>
                            <td>{metric_stats['max']:.4f}</td>
                            <td>{convergence['mean_epochs']:.1f} ± {convergence['std_epochs']:.1f}</td>
                            <td>{stability['mean_variance']:.2e} ± {stability['std_variance']:.2e}</td>
                        </tr>
                    """
                html += "</table></div>"
        html += "</div>"

        # Modified plots section for phase-split plots
        html += '<div class="plots-grid">'

        # Group by metric type
        for metric_type in k_fold_plots["by_phase"].keys():
            html += f"""
                <div class="card plot-section">
                    <h2>{metric_type.name.replace("_", " ").title()}</h2>
                    <div class="metrics-container">
                        <div class="metric-row">
            """

            # Show phase plots side by side
            for phase in ["training", "validation", "testing"]:
                if phase in k_fold_plots["by_phase"][metric_type]:
                    html += f"""
                        <div class="metric-item">
                            <h3>{phase.capitalize()}</h3>
                            <img src="{k_fold_plots["by_phase"][metric_type][phase]}" 
                                alt="{phase} {metric_type.name}">
                        </div>
                    """

            # Add aggregate and boxplot for this metric
            html += f"""
                        </div>
                        <div class="metric-row summary-plots">
                            <div class="metric-item">
                                <h3>Aggregate Statistics</h3>
                                <img src="{k_fold_plots["aggregate"][metric_type]}" 
                                    alt="Aggregate {metric_type.name}">
                            </div>
                            <div class="metric-item">
                                <h3>Distribution</h3>
                                <img src="{k_fold_plots["boxplots"][metric_type]}" 
                                    alt="Box plot {metric_type.name}">
                            </div>
                        </div>
                    </div>
                </div>
            """

        html += "</div>"
        return html

    def _compute_kfold_statistics(self, results_list: List[Results]) -> Dict:
        """Compute statistics across folds for each metric and phase."""
        stats = {
            phase: {"metrics": {}, "convergence": {}, "stability": {}}
            for phase in ["training", "validation", "testing"]
        }

        for phase in ["training", "validation", "testing"]:
            # Get final metrics for each fold
            final_metrics = {}
            for metric_type in self.metrictypes:
                final_metrics[metric_type] = []

            for results in results_list:
                phase_data = getattr(results, phase)
                if phase_data:
                    for metric_type in self.metrictypes:
                        value = phase_data[-1].get_metric(metric_type)
                        if value is not None:
                            final_metrics[metric_type].append(value)

            # Compute statistics for each metric
            for metric_type, values in final_metrics.items():
                if values:
                    stats[phase]["metrics"][metric_type] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "values": values,
                    }

                    # Compute convergence (epochs to best metric)
                    epochs_to_best = []
                    for results in results_list:
                        phase_data = getattr(results, phase)
                        if phase_data:
                            metric_values = [epoch.get_metric(metric_type) for epoch in phase_data]
                            best_epoch = (
                                np.argmax(metric_values)
                                if metric_type.name.startswith("ACCURACY")
                                else np.argmin(metric_values)
                            )
                            epochs_to_best.append(best_epoch + 1)

                    stats[phase]["convergence"][metric_type] = {
                        "mean_epochs": float(np.mean(epochs_to_best)),
                        "std_epochs": float(np.std(epochs_to_best)),
                    }

                    # Compute stability (variance in last few epochs)
                    last_epochs_variance = []
                    for results in results_list:
                        phase_data = getattr(results, phase)
                        if phase_data:
                            last_5_epochs = [
                                epoch.get_metric(metric_type) for epoch in phase_data[-5:]
                            ]
                            last_epochs_variance.append(np.var(last_5_epochs))

                    stats[phase]["stability"][metric_type] = {
                        "mean_variance": float(np.mean(last_epochs_variance)),
                        "std_variance": float(np.std(last_epochs_variance)),
                    }

        return stats

    def style_plot(self, title: str, xlabel: str = "Epoch"):
        """Helper to apply consistent theme styling to plots."""
        theme = self._get_theme_css()[self.color_scheme]

        plt.gca().set_facecolor(theme["bg-secondary"])
        plt.gcf().set_facecolor(theme["bg-primary"])

        plt.title(title, color=theme["text-primary"])
        plt.xlabel(xlabel, color=theme["text-secondary"])
        plt.ylabel(title.split()[-1], color=theme["text-secondary"])

        plt.grid(True, alpha=0.3, color=theme["border"])
        plt.tick_params(colors=theme["text-secondary"])

        # fix legend styling
        legend = plt.legend(
            facecolor=theme["bg-secondary"], edgecolor=theme["border"], framealpha=1.0
        )
        plt.setp(legend.get_texts(), color=theme["text-primary"])
        legend.get_frame().set_linewidth(1)
        legend.get_frame().set_alpha(1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    def _generate_kfold_plots(
        self, results_list: List[Results], k: int, plot_dir: str
    ) -> Dict[str, Dict[MetricType, str]]:
        plots_info = {"by_phase": {}, "aggregate": {}, "boxplots": {}}
        theme = self._get_theme_css()[self.color_scheme]
        colors = self._get_plot_colors(k)

        for metric_type in self.metrictypes:
            # skip confusion matrix metrics in main plots
            if not MetricPlotter.should_plot_separately(metric_type):
                continue

            plots_info["by_phase"][metric_type] = {}

            # 1. Phase-separated plots
            for phase in ["training", "validation", "testing"]:
                plt.figure(figsize=(12, 7))

                for fold_idx, results in enumerate(results_list):
                    metric_data = MetricPlotter.extract_metric_data(results, metric_type)
                    data = metric_data[phase]

                    if len(data) > 0:
                        plt.plot(
                            data, label=f"Fold {fold_idx + 1}", color=colors[fold_idx], alpha=0.8
                        )

                self.style_plot(
                    f'{phase.capitalize()} {metric_type.name.replace("_", " ").title()}'
                )
                plot_name = f"{phase}_{metric_type.name.lower()}.png"
                plot_path = os.path.join(plot_dir, plot_name)
                plt.savefig(plot_path, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
                plt.close()
                plots_info["by_phase"][metric_type][phase] = os.path.relpath(
                    plot_path, self.run_dir
                )

            # 2. Aggregate statistics plot
            plt.figure(figsize=(12, 7))

            for phase in ["training", "validation", "testing"]:
                epoch_data = []
                for results in results_list:
                    phase_data = getattr(results, phase)
                    if phase_data:
                        # handle integer metrics properly
                        convert_fn = (
                            int if metric_type in MetricPlotter.CONFUSION_METRICS else float
                        )
                        metrics = [
                            convert_fn(epoch.get_metric(metric_type)) for epoch in phase_data
                        ]
                        epoch_data.append(metrics)

                if epoch_data:
                    max_epochs = max(len(data) for data in epoch_data)
                    # use appropriate fill value based on metric type
                    fill_value = 0 if metric_type in MetricPlotter.CONFUSION_METRICS else np.nan
                    padded_data = [
                        np.pad(
                            data,
                            (0, max_epochs - len(data)),
                            "constant",
                            constant_values=fill_value,
                        )
                        for data in epoch_data
                    ]

                    # use nanmean/nanstd for float metrics, regular mean/std for integers
                    if metric_type in MetricPlotter.CONFUSION_METRICS:
                        mean_curve = np.mean(padded_data, axis=0)
                        std_curve = np.std(padded_data, axis=0)
                    else:
                        mean_curve = np.nanmean(padded_data, axis=0)
                        std_curve = np.nanstd(padded_data, axis=0)

                    epochs = np.arange(1, max_epochs + 1)
                    plt.plot(
                        epochs,
                        mean_curve,
                        label=f"{phase.capitalize()} (mean)",
                        color=MetricPlotter.PHASE_COLORS[phase],
                    )
                    plt.fill_between(
                        epochs,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2,
                        color=MetricPlotter.PHASE_COLORS[phase],
                    )

            self.style_plot(f'{metric_type.name.replace("_", " ").title()} (Mean ± Std)')
            plot_name = f"aggregate_{metric_type.name.lower()}.png"
            plot_path = os.path.join(plot_dir, plot_name)
            plt.savefig(plot_path, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
            plt.close()
            plots_info["aggregate"][metric_type] = os.path.relpath(plot_path, self.run_dir)

            # 3. Box plots (similar changes for integer handling)
            plt.figure(figsize=(10, 6))
            final_metrics = {phase: [] for phase in ["training", "validation", "testing"]}

            for results in results_list:
                for phase in ["training", "validation", "testing"]:
                    phase_data = getattr(results, phase)
                    if phase_data:
                        convert_fn = (
                            int if metric_type in MetricPlotter.CONFUSION_METRICS else float
                        )
                        final_metrics[phase].append(
                            convert_fn(phase_data[-1].get_metric(metric_type))
                        )

            data = [values for values in final_metrics.values() if values]
            labels = [phase.capitalize() for phase, values in final_metrics.items() if values]

            boxplot = plt.boxplot(data, labels=labels, patch_artist=True)
            for i, box in enumerate(boxplot["boxes"]):
                box_color = MetricPlotter.PHASE_COLORS[["training", "validation", "testing"][i]]
                box.set(facecolor=box_color, alpha=0.3)
                plt.setp(boxplot["medians"][i], color=box_color)
                plt.setp(boxplot["whiskers"][i * 2 : i * 2 + 2], color=box_color)
                plt.setp(boxplot["caps"][i * 2 : i * 2 + 2], color=box_color)
                plt.setp(boxplot["fliers"][i], markeredgecolor=box_color)

            self.style_plot(
                f'Final {metric_type.name.replace("_", " ").title()} Distribution', xlabel="Phase"
            )
            plot_name = f"boxplot_{metric_type.name.lower()}.png"
            plot_path = os.path.join(plot_dir, plot_name)
            plt.savefig(plot_path, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
            plt.close()
            plots_info["boxplots"][metric_type] = os.path.relpath(plot_path, self.run_dir)

        # Add confusion matrices for each fold's final epoch
        for fold_idx, results in enumerate(results_list):
            for phase in ["training", "validation"]:
                phase_data = getattr(results, phase)
                if phase_data:
                    # get confusion matrix for this fold/phase
                    cm = MetricPlotter.get_confusion_matrix_for_phase(results, phase)

                    # plot it
                    fig = MetricPlotter.plot_confusion_matrix(
                        cm, f"{phase.capitalize()} Confusion Matrix (Fold {fold_idx+1})", theme
                    )

                    # save it
                    plot_name = f"confusion_matrix_{phase}_fold_{fold_idx+1}.png"
                    plot_path = os.path.join(plot_dir, plot_name)
                    fig.savefig(plot_path, bbox_inches="tight", facecolor=fig.get_facecolor())
                    plt.close(fig)

                    if "confusion" not in plots_info:
                        plots_info["confusion"] = []
                    plots_info["confusion"].append(
                        (fold_idx, phase, os.path.relpath(plot_path, self.run_dir))
                    )

        return plots_info

    def _get_plot_colors(self, k: int) -> List[str]:
        """Generate k distinct colors from the current theme."""
        theme = self._get_theme_css()[self.color_scheme]

        # base colors from theme
        base_colors = [
            theme["accent"],
            theme["text-primary"],
            theme["text-secondary"],
            theme["border"],
        ]

        # create variations of accent color for more options
        # shifting hue while keeping similar saturation/value

        def hex_to_hsv(hex_color: str) -> tuple:
            # convert hex to rgb
            hex_color = hex_color.lstrip("#")
            rgb = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
            return colorsys.rgb_to_hsv(*rgb)

        def hsv_to_hex(hsv: tuple) -> str:
            rgb = colorsys.hsv_to_rgb(*hsv)
            return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

        # generate variations of accent color
        accent_hsv = hex_to_hsv(theme["accent"])
        variations = []
        for i in range(k - len(base_colors)):
            new_hue = (accent_hsv[0] + (i + 1) * (1.0 / k)) % 1.0
            variations.append(hsv_to_hex((new_hue, accent_hsv[1], accent_hsv[2])))

        all_colors = base_colors + variations

        # ensure we have k colors by cycling if needed
        return (
            all_colors[:k]
            if len(all_colors) >= k
            else (all_colors * (k // len(all_colors) + 1))[:k]
        )

        def _generate_confusion_matrices(self, results: Results) -> Dict[str, str]:
            """Generate confusion matrix plots for single training run"""
            confusion_plots = {}
            theme = self._get_theme_css()[self.color_scheme]

            for phase in ["training", "validation"]:
                if hasattr(results, phase) and getattr(results, phase):
                    cm = MetricPlotter.get_confusion_matrix_for_phase(results, phase)

                    fig = MetricPlotter.plot_confusion_matrix(
                        cm, f"{phase.capitalize()} Confusion Matrix", theme
                    )

                    plot_name = f"confusion_matrix_{phase}_final.png"
                    plot_path = os.path.join(self.plot_dir, plot_name)
                    fig.savefig(plot_path, bbox_inches="tight", facecolor=fig.get_facecolor())
                    plt.close(fig)

                    confusion_plots[phase] = os.path.relpath(plot_path, self.run_dir)

            return confusion_plots

    def _generate_kfold_confusion_matrices(self, k_fold_results: List[Results]) -> Dict[str, str]:
        """Generate averaged confusion matrix plots for k-fold results"""
        confusion_plots = {"averaged": {}, "individual": []}
        theme = self._get_theme_css()[self.color_scheme]

        # Generate averaged matrices
        for phase in ["training", "validation"]:
            avg_cm = MetricPlotter.get_average_confusion_matrix(k_fold_results, phase)

            fig = MetricPlotter.plot_confusion_matrix(
                avg_cm, f"{phase.capitalize()} Confusion Matrix (Averaged)", theme
            )

            plot_name = f"confusion_matrix_{phase}_avg.png"
            plot_path = os.path.join(self.plot_dir, plot_name)
            fig.savefig(plot_path, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            confusion_plots["averaged"][phase] = os.path.relpath(plot_path, self.run_dir)

        # Generate individual fold matrices
        for fold_idx, results in enumerate(k_fold_results):
            for phase in ["training", "validation"]:
                if hasattr(results, phase) and getattr(results, phase):
                    cm = MetricPlotter.get_confusion_matrix_for_phase(results, phase)

                    fig = MetricPlotter.plot_confusion_matrix(
                        cm, f"{phase.capitalize()} Confusion Matrix (Fold {fold_idx + 1})", theme
                    )

                    plot_name = f"confusion_matrix_{phase}_fold_{fold_idx + 1}.png"
                    plot_path = os.path.join(self.plot_dir, plot_name)
                    fig.savefig(plot_path, bbox_inches="tight", facecolor=fig.get_facecolor())
                    plt.close(fig)

                    confusion_plots["individual"].append(
                        (fold_idx, phase, os.path.relpath(plot_path, self.run_dir))
                    )

        return confusion_plots

    def _get_confusion_section_html(
        self, confusion_plots: Dict[str, str], is_kfold: bool = False
    ) -> str:
        """Generate HTML section for confusion matrix visualization"""
        if not is_kfold:
            # Regular non-k-fold confusion matrix display (unchanged)
            title = "Final Confusion Matrices"
            html = f"""
                <div class='card'>
                    <h2>{title}</h2>
                    <div class='confusion-grid'>
            """

            for phase, plot_path in confusion_plots.items():
                html += f"""
                    <div class='confusion-matrix'>
                        <h3>{phase.capitalize()} Phase</h3>
                        <img src='{plot_path}' alt='{phase} confusion matrix'>
                    </div>
                """

            html += "</div></div>"
            return html

        else:
            # K-fold version
            html = """
                <div class='card'>
                    <h2>Confusion Matrices</h2>
                    
                    <!-- Averaged matrices -->
                    <h3>Averaged Across Folds</h3>
                    <div class='confusion-grid'>
            """

            # Add averaged matrices
            for phase, plot_path in confusion_plots["averaged"].items():
                html += f"""
                    <div class='confusion-matrix'>
                        <h3>{phase.capitalize()} Phase</h3>
                        <img src='{plot_path}' alt='{phase} average confusion matrix'>
                    </div>
                """

            # Add individual fold matrices in paired columns
            html += """
                </div>
                
                <h3 class='fold-matrices-title'>Individual Fold Results</h3>
                <div class='fold-pairs'>
                    <div class='fold-header'>
                        <div>Training Phase</div>
                        <div>Validation Phase</div>
                    </div>
            """

            # Group plots by fold
            fold_dict = {}
            for fold_idx, phase, plot_path in confusion_plots["individual"]:
                if fold_idx not in fold_dict:
                    fold_dict[fold_idx] = {}
                fold_dict[fold_idx][phase] = plot_path

            # Generate rows for each fold
            for fold_idx in sorted(fold_dict.keys()):
                html += f"""
                    <div class='fold-row'>
                        <h4 class='fold-label'>Fold {fold_idx + 1}</h4>
                        <div class='fold-pair'>
                            <div class='confusion-matrix'>
                                <img src='{fold_dict[fold_idx]["training"]}' 
                                    alt='fold {fold_idx + 1} training confusion matrix'>
                            </div>
                            <div class='confusion-matrix'>
                                <img src='{fold_dict[fold_idx]["validation"]}' 
                                    alt='fold {fold_idx + 1} validation confusion matrix'>
                            </div>
                        </div>
                    </div>
                """

            html += "</div></div>"
            return html

    def _get_theme_css(self) -> str:
        themes = {
            ColorScheme.LIGHT: {
                "bg-primary": "#fafafa",
                "bg-secondary": "#ffffff",
                "text-primary": "#2c3e50",
                "text-secondary": "#666666",
                "border": "#dddddd",
                "shadow": "rgba(0,0,0,0.1)",
                "accent": "#3498db",
                "accent-subtle": "#f7f9fc",
            },
            ColorScheme.DARK: {
                "bg-primary": "#1a1a1a",
                "bg-secondary": "#2d2d2d",
                "text-primary": "#e0e0e0",
                "text-secondary": "#a0a0a0",
                "border": "#404040",
                "shadow": "rgba(0,0,0,0.3)",
                "accent": "#4a9eff",
                "accent-subtle": "#2d3436",
            },
            ColorScheme.SYNTHWAVE: {
                "bg-primary": "#2b1f38",
                "bg-secondary": "#33244a",
                "text-primary": "#ff7edb",
                "text-secondary": "#e079c3",
                "border": "#553d67",
                "shadow": "rgba(255,126,219,0.1)",
                "accent": "#f97e72",
                "accent-subtle": "#342b4a",
            },
            ColorScheme.KANAGAWA: {  # based on kanagawa.nvim
                "bg-primary": "#1F1F28",
                "bg-secondary": "#2A2A37",
                "text-primary": "#DCD7BA",
                "text-secondary": "#C8C093",
                "border": "#363646",
                "shadow": "rgba(0,0,0,0.3)",
                "accent": "#7E9CD8",
                "accent-subtle": "#363646",
            },
            ColorScheme.CATPPUCCIN_MOCHA: {
                "bg-primary": "#1e1e2e",
                "bg-secondary": "#313244",
                "text-primary": "#cdd6f4",
                "text-secondary": "#bac2de",
                "border": "#45475a",
                "shadow": "rgba(0,0,0,0.3)",
                "accent": "#89b4fa",
                "accent-subtle": "#45475a",
            },
            ColorScheme.CATPPUCCIN_MACCHIATO: {
                "bg-primary": "#24273a",
                "bg-secondary": "#363a4f",
                "text-primary": "#cad3f5",
                "text-secondary": "#b8c0e0",
                "border": "#494d64",
                "shadow": "rgba(0,0,0,0.3)",
                "accent": "#8aadf4",
                "accent-subtle": "#494d64",
            },
            ColorScheme.IBM_CARBON: {
                "bg-primary": "#161616",
                "bg-secondary": "#262626",
                "text-primary": "#f4f4f4",
                "text-secondary": "#c6c6c6",
                "border": "#393939",
                "shadow": "rgba(0,0,0,0.4)",
                "accent": "#78a9ff",
                "accent-subtle": "#2d2d2d",
            },
            ColorScheme.FRIEREN: {  # based on her color scheme/magic
                "bg-primary": "#2a2438",  # deep purple like her cloak
                "bg-secondary": "#352f44",  # slightly lighter purple
                "text-primary": "#e9e9e9",  # silver like her hair
                "text-secondary": "#b9b4c7",  # muted silver
                "border": "#5c5470",  # dusty purple
                "shadow": "rgba(42,36,56,0.3)",  # shadow matching bg
                "accent": "#9ad0c2",  # mint green for her magic
                "accent-subtle": "#2d2a3b",  # deeper purple for subtle effects
            },
            ColorScheme.DRACULA: {
                "bg-primary": "#282a36",
                "bg-secondary": "#44475a",
                "text-primary": "#f8f8f2",
                "text-secondary": "#6272a4",
                "border": "#44475a",
                "shadow": "rgba(0,0,0,0.3)",
                "accent": "#bd93f9",
                "accent-subtle": "#373844",
            },
            ColorScheme.GRUVBOX: {
                "bg-primary": "#282828",
                "bg-secondary": "#3c3836",
                "text-primary": "#ebdbb2",
                "text-secondary": "#a89984",
                "border": "#504945",
                "shadow": "rgba(0,0,0,0.3)",
                "accent": "#b8bb26",
                "accent-subtle": "#32302f",
            },
            ColorScheme.NORD: {
                "bg-primary": "#2e3440",
                "bg-secondary": "#3b4252",
                "text-primary": "#eceff4",
                "text-secondary": "#d8dee9",
                "border": "#434c5e",
                "shadow": "rgba(0,0,0,0.3)",
                "accent": "#88c0d0",
                "accent-subtle": "#434c5e",
            },
        }
        return themes

    def _get_css(self) -> str:
        current_theme = self._get_theme_css()[self.color_scheme]

        theme_css = f"""
            /* theme variables */
            :root {{
                --bg-primary: {current_theme["bg-primary"]};
                --bg-secondary: {current_theme["bg-secondary"]};
                --text-primary: {current_theme["text-primary"]};
                --text-secondary: {current_theme["text-secondary"]};
                --border: {current_theme["border"]};
                --shadow: {current_theme["shadow"]};
                --accent: {current_theme["accent"]};
                --accent-subtle: {current_theme["accent-subtle"]};
            }}

            /* base styles using theme variables */
            body {{
                font-family: -apple-system, system-ui, BlinkMacSystemFont;
                max-width: 1600px;
                margin: 0 auto;
                padding: 20px;
                background: var(--bg-primary);
                color: var(--text-primary);
            }}
            
            .card, .config-card {{
                background: var(--bg-secondary);
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px var(--shadow);
            }}
            
            .config-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid var(--border);
            }}
            
            th {{
                color: var(--accent);
                background: var(--accent-subtle);
            }}
            
            h1, h2, h3 {{
                color: var(--text-primary);
                margin-top: 0;
            }}
            
            .phase-header {{
                color: var(--text-secondary);
                font-size: 1.2em;
                margin-top: 20px;
            }}
            
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                border: 1px solid var(--border);
            }}
            
            .notes {{
                margin-top: 20px;
                padding: 15px;
                background: var(--accent-subtle);
                border-radius: 4px;
            }}
            
            /* some subtle hover effects */
            .config-card:hover {{
                box-shadow: 0 4px 8px var(--shadow);
                transition: box-shadow 0.3s ease;
            }}
            
            tr:hover {{
                background: var(--accent-subtle);
                transition: background 0.2s ease;
            }}
        """

        plot_css = """
            .plots-grid {
                display: flex;
                flex-direction: column;
                gap: 30px;
                margin: 20px 0;
            }

            .plot-section {
                margin-bottom: 20px;
            }

            .metrics-container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }

            .metric-row {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                width: 100%;
            }

            .summary-plots {
                grid-template-columns: repeat(2, 1fr);
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid var(--border);
            }

            .metric-item {
                display: flex;
                flex-direction: column;
                gap: 10px;
                width: 100%;
            }

            .metric-item h3 {
                margin: 0;
                font-size: 1.1em;
                color: var(--accent);
                text-align: center;
            }

            .metric-item img {
                width: 100%;
                height: auto;
                object-fit: contain;
            }

            /* Responsive adjustments */
            @media (max-width: 1400px) {
                .metric-row {
                    grid-template-columns: repeat(2, 1fr);
                }
            }

            @media (max-width: 900px) {
                .metric-row {
                    grid-template-columns: 1fr;
                }
            }
        """

        confusion_css = """
            .confusion-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
                align-items: start;
            }
            
            .fold-matrices-title {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid var(--border);
            }
            
            .fold-pairs {
                margin: 20px 0;
            }
            
            .fold-header {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
                text-align: center;
                font-weight: bold;
                color: var(--accent);
            }
            
            .fold-row {
                margin-bottom: 30px;
            }
            
            .fold-label {
                color: var(--text-secondary);
                margin: 0 0 10px 0;
                text-align: center;
            }
            
            .fold-pair {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                align-items: center;
            }
            
            .confusion-matrix {
                text-align: center;
            }
            
            .confusion-matrix h3 {
                margin-bottom: 10px;
                color: var(--accent);
            }
            
            .confusion-matrix img {
                max-width: 100%;
                height: auto;
                margin: 0 auto;
            }
            
            /* make it stack on mobile */
            @media (max-width: 768px) {
                .confusion-grid, .fold-pair, .fold-header {
                    grid-template-columns: 1fr;
                }
                
                .fold-header div:last-child {
                    margin-top: -15px;  /* reduce space between headers when stacked */
                }
            }
        """

        return theme_css + plot_css + confusion_css

    def _serialize_results(self, results: Results) -> Dict[str, Any]:
        return {
            phase: [
                {k.name: v for k, v in epoch.metrics.items()} for epoch in getattr(results, phase)
            ]
            for phase in ["training", "validation", "testing"]
        }

    def _serialize_config(self) -> Dict[str, Any]:
        return {
            k: str(v) if isinstance(v, (type, torch.device)) else v
            for k, v in self.config.__dict__.items()
        }

    def save_run_data(self, results: Results):
        data = {
            "timestamp": self.timestamp,
            "config": self._serialize_config(),
            "system_info": self.system_info,
            "results": self._serialize_results(results),
        }

        with open(os.path.join(self.run_dir, "run_data.json"), "w") as f:
            json.dump(data, f, indent=2)

    def _save_kfold_data(self, results_list: List[Results]):
        """
        Save k-fold validation data similar to save_run_data but with k-fold context.
        Reuses existing serialization methods for consistency.
        """
        k = len(results_list)

        data = {
            "timestamp": self.timestamp,
            "config": self._serialize_config(),
            "system_info": self.system_info,
            "k_fold_context": {
                "num_folds": k,
                "total_samples": self.dataset_info.training_size
                + self.dataset_info.validation_size
                + self.dataset_info.testing_size,
                "fold_sizes": {
                    "training": self.dataset_info.training_size // k,
                    "validation": self.dataset_info.validation_size // k
                    if self.dataset_info.validation_size
                    else 0,
                    "testing": self.dataset_info.testing_size // k
                    if self.dataset_info.testing_size
                    else 0,
                },
            },
            "folds": [
                {"fold_index": i, "results": self._serialize_results(results)}
                for i, results in enumerate(results_list)
            ],
        }

        with open(os.path.join(self.run_dir, "kfold_data.json"), "w") as f:
            json.dump(data, f, indent=2)

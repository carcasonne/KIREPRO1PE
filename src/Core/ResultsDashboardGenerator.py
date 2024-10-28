import os
import platform
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import psutil
import torch

from Core.Metrics import MetricType, Results


@dataclass
class ExperimentConfig:
    # core training params
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str

    # model architecture
    model_name: str
    hidden_dims: List[int]
    activation: str

    # regularization
    dropout: float = 0.0
    weight_decay: float = 0.0

    # misc
    random_seed: Optional[int] = None
    notes: str = ""


class MetricPlotter:
    PHASE_COLORS = {
        "training": "#2ecc71",  # nice green
        "validation": "#3498db",  # nice blue
        "testing": "#e74c3c",  # nice red
    }

    @staticmethod
    def extract_metric_data(results: Results, metric_type: MetricType) -> Dict[str, List[float]]:
        return {
            "training": [epoch.get_metric(metric_type) for epoch in results.training],
            "validation": [epoch.get_metric(metric_type) for epoch in results.validation],
            "testing": [epoch.get_metric(metric_type) for epoch in results.testing],
        }


class TrainingReporter:
    def __init__(self, config: ExperimentConfig, base_dir: str = "training_runs"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, self.timestamp)
        self.plot_dir = os.path.join(self.run_dir, "plots")
        self.config = config

        # get system info
        self.system_info = {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "torch_version": torch.__version__,
            "ram": f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }

        os.makedirs(self.plot_dir, exist_ok=True)

    def generate_report(self, results: Results) -> str:
        """generates plots and dashboard for a training run"""
        plots_info = self._generate_plots(results)
        return self._generate_dashboard(plots_info, results)

    def _generate_plots(self, results: Results) -> Dict[MetricType, str]:
        plots_info = {}

        for metric_type in MetricType:
            metric_data = MetricPlotter.extract_metric_data(results, metric_type)

            # skip empty metrics
            if not any(len(data) > 0 for data in metric_data.values()):
                continue

            plt.figure(figsize=(10, 6))

            for phase, data in metric_data.items():
                if len(data) > 0:  # only plot if we have data
                    plt.plot(
                        data,
                        label=phase.capitalize(),
                        color=MetricPlotter.PHASE_COLORS[phase],
                        alpha=0.8,
                    )

            plt.title(f'{metric_type.name.replace("_", " ").title()} Over Epochs')
            plt.xlabel("Epoch")
            plt.ylabel(metric_type.name.replace("_", " ").title())
            plt.grid(True, alpha=0.3)
            plt.legend()

            # save plot
            plot_name = f"{metric_type.name.lower()}.png"
            plot_path = os.path.join(self.plot_dir, plot_name)
            plt.savefig(plot_path)
            plt.close()

            plots_info[metric_type] = os.path.relpath(plot_path, self.run_dir)

        return plots_info

    def _generate_dashboard(self, plots_info: Dict[MetricType, str], results: Results) -> str:
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Run {self.timestamp}</title>
            <style>
                body {{
                    font-family: -apple-system, system-ui, BlinkMacSystemFont;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #fafafa;
                }}
                .card {{
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .phase-header {{
                    color: #666;
                    font-size: 1.2em;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Training Run {self.timestamp}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="card">
                <h2>Experiment Configuration</h2>
                <table>
                    <tr><th colspan="2">Training Parameters</th></tr>
                    <tr><td>Learning Rate</td><td>{self.config.learning_rate}</td></tr>
                    <tr><td>Batch Size</td><td>{self.config.batch_size}</td></tr>
                    <tr><td>Batches Shuffled</td><td>{self.config.shuffle_batches}</td></tr>
                    <tr><td>Epochs</td><td>{self.config.epochs}</td></tr>
                    <tr><td>Optimizer</td><td>{self.config.optimizer}</td></tr>
                    <tr><td>Torch Seed</td><td>{self.config.torch_seed}</td></tr>
                    
                    <tr><th colspan="2">Model Architecture</th></tr>
                    <tr><td>Model</td><td>{self.config.model_name}</td></tr>
                    <tr><td>Hidden Dims</td><td>{self.config.hidden_dims}</td></tr>
                    <tr><td>Activation</td><td>{self.config.activation}</td></tr>
                    
                    <tr><th colspan="2">Regularization</th></tr>
                    <tr><td>Dropout</td><td>{self.config.dropout}</td></tr>
                    <tr><td>Weight Decay</td><td>{self.config.weight_decay}</td></tr>

                    <tr><th colspan="2">Data Specifications</th></tr>
                    <tr><td>Audio Sample Rate</td><td>{self.config.sample_rate}</td></tr>
                    <tr><td>Audio Duration</td><td>{self.config.duration}</td></tr>
                    <tr><td>Spectogram Image Height</td><td>{self.config.img_height}</td></tr>
                    <tr><td>Spectogram Image Width</td><td>{self.config.img_width}</td></tr>
                    <tr><td>Spectogram Image Channels</td><td>{self.config.channels}</td></tr>
                    
                    <tr><th colspan="2">System Info</th></tr>
                    {"".join(f"<tr><td>{k.replace('_', ' ').title()}</td><td>{v}</td></tr>" for k, v in self.system_info.items())}
                </table>
                
                {f'<div class="notes"><h3>Notes</h3><p>{self.config.notes}</p></div>' if self.config.notes else ''}
            </div>
        """

        # plots section
        html += '<div class="card"><h2>Training Metrics</h2><div class="metric-grid">'
        for metric_type, plot_path in plots_info.items():
            html += f"""
                <div>
                    <h3>{metric_type.name.replace("_", " ").title()}</h3>
                    <img src="{plot_path}" alt="{metric_type.name}">
                </div>
            """
        html += "</div></div>"

        # final metrics tables
        html += '<div class="card"><h2>Final Metrics</h2>'

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
                        </tr>
                    """
                html += "</table>"

        html += "</div></body></html>"

        output_path = os.path.join(self.run_dir, "report.html")
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

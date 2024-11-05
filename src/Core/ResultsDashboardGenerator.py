import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import psutil
import torch
from torch.utils.data import DataLoader

from config import AudioClassifierConfig
from Core.Metrics import MetricType, Results


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


class MetricPlotter:
    PHASE_COLORS = {
        "training": "#2ecc71",  # like Frieren's cloak
        "validation": "#3498db",  # stark's magic
        "testing": "#e74c3c",  # demon lord vibes
    }

    @staticmethod
    def extract_metric_data(results: Results, metric_type: MetricType) -> Dict[str, List[float]]:
        return {
            "training": [epoch.get_metric(metric_type) for epoch in results.training],
            "validation": [epoch.get_metric(metric_type) for epoch in results.validation],
            "testing": [epoch.get_metric(metric_type) for epoch in results.testing],
        }


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
        )

        os.makedirs(self.plot_dir, exist_ok=True)

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

    def generate_report(self, results: Results) -> str:
        plots_info = self._generate_plots(results)
        self.save_run_data(results)
        return self._generate_dashboard(plots_info, results)

    def _generate_plots(self, results: Results) -> Dict[MetricType, str]:
        # existing plot generation code remains the same
        plots_info = {}
        for metric_type in MetricType:
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
            plt.title(f'{metric_type.name.replace("_", " ").title()} Over Epochs')
            plt.xlabel("Epoch")
            plt.ylabel(metric_type.name.replace("_", " ").title())
            plt.grid(True, alpha=0.3)
            plt.legend()
            plot_name = f"{metric_type.name.lower()}.png"
            plot_path = os.path.join(self.plot_dir, plot_name)
            plt.savefig(plot_path)
            plt.close()
            plots_info[metric_type] = os.path.relpath(plot_path, self.run_dir)
        return plots_info

    def _generate_dashboard(self, plots_info: Dict[MetricType, str], results: Results) -> str:
        dataloader_section = f"""
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

        run_config_section = f"""
            <div class="config-card">
                <h2>Run Configuration</h2>
                <table>
                    <tr><th colspan="2">Training Parameters</th></tr>
                    <tr><td>Learning Rate</td><td>{self.config.learning_rate}</td></tr>
                    <tr><td>Optimizer</td><td>{self.config.optimizer}</td></tr>
                    <tr><td>Torch Seed</td><td>{self.config.torch_seed}</td></tr>
                    
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
                <p>Color Scheme: {self.color_scheme}</p>
            </div>

            <div class="config-grid">
                {dataloader_section}
                {run_config_section}
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

        # metrics tables
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
        return theme_css

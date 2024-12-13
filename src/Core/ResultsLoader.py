import json
from dataclasses import dataclass
from typing import List, Union

from Core.Metrics import EpochMetrics, MetricType, Results


@dataclass
class ResultsLoader:
    @staticmethod
    def load(json_path: str) -> Union[Results, List[Results]]:
        with open(json_path, "r") as f:
            data = json.load(f)

        def _json_to_metrics(metrics_list: List[dict]) -> List[EpochMetrics]:
            return [
                EpochMetrics(metrics={MetricType[k]: v for k, v in epoch.items()})
                for epoch in metrics_list
            ]

        if "results" in data:
            results = Results()
            for phase in ["training", "validation", "testing"]:
                if phase in data["results"]:
                    setattr(results, phase, _json_to_metrics(data["results"][phase]))
            return results
        elif "folds" in data:
            return [
                Results(
                    training=_json_to_metrics(fold["results"]["training"]),
                    validation=_json_to_metrics(fold["results"]["validation"]),
                    testing=_json_to_metrics(fold["results"]["testing"]),
                )
                for fold in data["folds"]
            ]

        raise ValueError("json format looking kinda sus")

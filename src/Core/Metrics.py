from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict


class MetricType(Enum):
    ABSOLUTE_LOSS = auto()
    AVERAGE_LOSS = auto()
    ACCURACY = auto()


# This allows for expanding MetricType easily later without any dependencies breaking
# when plotting do something like: losses = [epoch.get_metric(MetricType.ABSOLUTE_LOSS) for epoch in results.training]
@dataclass
class EpochMetrics:
    metrics: Dict[MetricType, float] = field(default_factory=dict)

    def add_metric(self, metric_type: MetricType, value: float):
        self.metrics[metric_type] = value

    def get_metric(self, metric_type: MetricType) -> float:
        return self.metrics.get(metric_type, 0.0)


@dataclass
class Results:
    training: list[EpochMetrics] = field(default_factory=list)
    validation: list[EpochMetrics] = field(default_factory=list)
    testing: list[EpochMetrics] = field(default_factory=list)

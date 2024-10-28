from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List


# Here define all the metrics we care about
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


# field(default_factory=list) makes a new list for each instance of this DataClass
# crazy behaviour that that is not default
## TODO: Maybe we want to split this by batches too later, we're probabably gonna have a shitton of input
@dataclass
class Results:
    training: List[EpochMetrics] = field(default_factory=list)
    validation: List[EpochMetrics] = field(default_factory=list)
    testing: List[EpochMetrics] = field(default_factory=list)

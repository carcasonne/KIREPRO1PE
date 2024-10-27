from dataclasses import dataclass, field
from typing import List

from Core.Metrics import EpochMetrics


# field(default_factory=list) makes a new list for each instance of this DataClass
# crazy behaviour that that is not default
## TODO: Maybe we want to split this by batches too later, we're probabably gonna have a shitton of input
@dataclass
class Results:
    training: List[EpochMetrics] = field(default_factory=list)
    validation: List[EpochMetrics] = field(default_factory=list)
    testing: List[EpochMetrics] = field(default_factory=list)

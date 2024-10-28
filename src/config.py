from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AudioClassifierConfig:
    # core training params
    learning_rate: float
    batch_size: int
    shuffle_batches: bool
    epochs: int
    optimizer: str

    # model architecture
    model_name: str
    hidden_dims: List[int]
    activation: str

    # classifier data specs
    sample_rate: int
    img_height: int
    img_width: int
    channels: int
    duration: int

    # misc
    data_path: str
    output_path: str
    run_cuda: bool
    notes: str = ""
    torch_seed: Optional[int] = None

    # regularization
    dropout: float = 0.0
    weight_decay: float = 0.0

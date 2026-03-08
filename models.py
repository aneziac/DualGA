from dataclasses import dataclass
from typing import Literal
import torch
import numpy as np
import numpy.typing as npt


@dataclass
class SRLConfig:
    gamma: float
    delta: float
    label: str


@dataclass
class DualGAConfig:
    gamma: float
    D: float
    init_lambda: float
    autoeta: float
    label: str


@dataclass
class ExperimentResult:
    label: str
    type: Literal["SRL", "DualGA"]
    z_scores: list[float]
    mean_dgs: list[float]
    mean_kls: list[float]


WatermarkResult = tuple[torch.Tensor, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]

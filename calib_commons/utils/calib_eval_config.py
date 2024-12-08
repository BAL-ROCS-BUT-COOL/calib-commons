from dataclasses import dataclass
from enum import Enum 

@dataclass
class CalibEvaluatorConfig:
    """Configuration defining the parameters use in the external calibrator."""
    reprojection_error_threshold: float = 1
    min_track_length: int = 2

    ba_least_square_ftol: float = 1e-8
    display: bool = False

"""
ScanTwin Configuration Module
=============================
Defines the CT reference protocol constants (kept for backwards compatibility; every other
modality defines its own reference in ``modalities/``), default criteria, and the research
disclaimer.

DISCLAIMER:
Research and educational prototype. ScanTwin uses simplified image-acquisition and
exposure models and is not intended for clinical diagnosis, patient-specific
dosimetry, or scanner/protocol prescription.
"""

from dataclasses import dataclass
from typing import Tuple, List

# Mandatory Research Disclaimer
RESEARCH_DISCLAIMER: str = (
    "Research and educational prototype. ScanTwin uses simplified image-acquisition and "
    "exposure models and is not intended for clinical diagnosis, patient-specific "
    "dosimetry, or scanner/protocol prescription."
)

# Reference Clinical-like Baseline Acquisition Protocol
REFERENCE_PROJECTIONS: int = 360
REFERENCE_I0: float = 1e5  # Relative incident photon budget proxy
REFERENCE_METHOD: str = "FBP"
REFERENCE_FILTER: str = "ramp"
REFERENCE_SART_ITERATIONS: int = 5

# Supported Reconstruction Filters for FBP
SUPPORTED_FBP_FILTERS: List[str] = ["ramp", "shepp-logan", "cosine", "hamming", "hann"]

# Supported Reconstruction Algorithms
SUPPORTED_METHODS: List[str] = ["FBP", "SART"]

# Default Phantom & Geometry Configuration
DEFAULT_IMAGE_SIZE: int = 256
DEFAULT_ANGULAR_RANGE_DEG: Tuple[float, float] = (0.0, 180.0)

# Default Synthetic Lesion Configuration
DEFAULT_LESION_RADIUS: float = 8.0  # pixels in 256x256 grid
DEFAULT_LESION_CONTRAST: float = 0.08  # relative contrast relative to local background
DEFAULT_LESION_CENTER: Tuple[int, int] = (164, 72)  # (row, col) inside homogeneous brain parenchyma

# Optimization Default Constraints & Bounds
DEFAULT_MIN_SSIM: float = 0.88
DEFAULT_MIN_CNR: float = 3.0
DEFAULT_MIN_PSNR: float = 28.0

# Pages / app wording
APP_SUBTITLE: str = (
    "An Intelligent Digital Twin for Exposure-Aware Optimization of "
    "Medical Image Acquisition and Reconstruction"
)

# Data-use warning shown wherever images are loaded
DATA_USE_WARNING: str = (
    "Use only public, de-identified or properly licensed images. ScanTwin ships no patient data and "
    "downloads nothing automatically."
)

NO_FEASIBLE_MESSAGE: str = "No feasible protocol found within the current search space."

PHOTON_BUDGET_MIN: float = 1e3
PHOTON_BUDGET_MAX: float = 1e6
PROJECTIONS_MIN: int = 30
PROJECTIONS_MAX: int = 360


@dataclass
class AcquisitionProtocol:
    """Dataclass describing an acquisition & reconstruction protocol."""
    photon_budget: float = REFERENCE_I0
    num_projections: int = REFERENCE_PROJECTIONS
    reconstruction_method: str = REFERENCE_METHOD
    fbp_filter: str = REFERENCE_FILTER
    sart_iterations: int = 1
    seed: int = 42

    @property
    def relative_photon_exposure(self) -> float:
        """Relative incident photon budget compared to reference baseline (percentage)."""
        return (self.photon_budget / REFERENCE_I0) * 100.0

    @property
    def projection_fraction(self) -> float:
        """Projection view count compared to reference baseline (percentage)."""
        return (self.num_projections / REFERENCE_PROJECTIONS) * 100.0

    @property
    def normalized_acquisition_cost(self) -> float:
        """
        Normalized acquisition-cost proxy:
        Combines relative exposure and projection fraction into a unitless cost proxy.
        C = 0.7 * (I0 / I0_ref) + 0.3 * (N / N_ref)
        Baseline protocol evaluates to 1.0.
        """
        return 0.7 * (self.photon_budget / REFERENCE_I0) + 0.3 * (self.num_projections / REFERENCE_PROJECTIONS)

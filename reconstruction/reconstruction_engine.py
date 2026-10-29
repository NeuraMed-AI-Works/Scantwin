"""
Unified Reconstruction Engine
=============================
Provides a standardized interface for analytical (FBP) and iterative (SART)
reconstruction algorithms.
"""

import time
from typing import Tuple, Dict, Any, Optional
import numpy as np

from reconstruction.fbp import reconstruct_fbp
from reconstruction.sart import reconstruct_sart


def reconstruct(
    sinogram: np.ndarray,
    theta: np.ndarray,
    method: str = "FBP",
    filter_name: str = "ramp",
    iterations: int = 1,
    circle: bool = True,
    relaxation: float = 0.15,
    output_size: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Unified entry point for CT reconstruction.

    Parameters
    ----------
    sinogram : np.ndarray
        2D projection data array.
    theta : np.ndarray
        1D array of projection angles in degrees.
    method : str, optional
        Reconstruction algorithm: "FBP" or "SART" (default is "FBP").
    filter_name : str, optional
        Filter name for FBP ("ramp", "shepp-logan", "cosine", "hamming", "hann").
    iterations : int, optional
        Number of iterations for SART (default is 1).
    circle : bool, optional
        Whether to mask outside the inscribed circular FOV.
    relaxation : float, optional
        SART relaxation parameter.
    output_size : int, optional
        Target spatial dimension.

    Returns
    -------
    reconstruction : np.ndarray
        2D reconstructed image array.
    metadata : dict
        Execution metrics including runtime and algorithm configuration.
    """
    method_upper = method.upper().strip()
    t_start = time.perf_counter()

    if method_upper == "FBP":
        recon = reconstruct_fbp(
            sinogram=sinogram,
            theta=theta,
            filter_name=filter_name,
            circle=circle,
            output_size=output_size,
        )
    elif method_upper == "SART":
        recon = reconstruct_sart(
            sinogram=sinogram,
            theta=theta,
            iterations=iterations,
            relaxation=relaxation,
        )
    else:
        raise ValueError(f"Reconstruction method '{method}' not recognized. Choose 'FBP' or 'SART'.")

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    metadata: Dict[str, Any] = {
        "method": method_upper,
        "filter_name": filter_name if method_upper == "FBP" else None,
        "iterations": iterations if method_upper == "SART" else None,
        "elapsed_time_ms": elapsed_ms,
        "min_val": float(np.min(recon)),
        "max_val": float(np.max(recon)),
        "mean_val": float(np.mean(recon)),
    }

    return recon, metadata

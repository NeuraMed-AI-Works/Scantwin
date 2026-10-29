"""
Simulation Package
==================
Modules for virtual phantom generation, synthetic lesion insertion,
forward projection (Radon transform), and photon noise modeling.
"""

from simulation.phantom import generate_shepp_logan, generate_disk_phantom
from simulation.lesion import insert_lesion
from simulation.projector import compute_radon_projection
from simulation.noise_model import apply_photon_noise
from simulation.acquisition import simulate_acquisition

__all__ = [
    "generate_shepp_logan",
    "generate_disk_phantom",
    "insert_lesion",
    "compute_radon_projection",
    "apply_photon_noise",
    "simulate_acquisition",
]

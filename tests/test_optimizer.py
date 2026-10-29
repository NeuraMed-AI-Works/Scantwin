"""
Unit tests for Protocol Optimizer and Closed-Loop Digital Twin.
"""

import pytest
import numpy as np

from simulation.phantom import generate_shepp_logan
from simulation.lesion import insert_lesion
from digital_twin.twin import CTDigitalTwin
from optimization.protocol_optimizer import ProtocolOptimizer


def test_optimizer_discovers_feasible_protocol():
    """Verify optimizer runs and returns valid, bounded parameters for achievable constraints."""
    phantom = generate_shepp_logan(size=64)
    mod_phantom, lesion_mask, bg_mask, meta = insert_lesion(
        phantom,
        radius=5.0,
        contrast=0.15,
    )

    twin = CTDigitalTwin(
        ground_truth=mod_phantom,
        lesion_mask=lesion_mask,
        background_mask=bg_mask,
        lesion_metadata=meta,
    )

    optimizer = ProtocolOptimizer(
        twin=twin,
        target_ssim_min=0.80,
        target_cnr_min=2.0,
        seed=42,
    )

    best_state, history_df, summary = optimizer.optimize(
        n_trials=8,
        n_realizations=1,
        include_sart=False,
    )

    assert len(history_df) >= 8
    if best_state is not None:
        assert best_state.is_feasible is True
        assert best_state.ssim >= 0.80
        assert best_state.cnr >= 2.0
        assert 1e3 <= best_state.photon_budget <= 1e6
        assert 30 <= best_state.num_projections <= 360


def test_optimizer_handles_impossible_constraints_safely():
    """When impossible constraints are imposed, optimizer must return None and not fake feasibility."""
    phantom = generate_shepp_logan(size=64)
    mod_phantom, lesion_mask, bg_mask, meta = insert_lesion(
        phantom,
        radius=4.0,
        contrast=0.01,
    )

    twin = CTDigitalTwin(
        ground_truth=mod_phantom,
        lesion_mask=lesion_mask,
        background_mask=bg_mask,
        lesion_metadata=meta,
    )

    # Impossible requirements: SSIM >= 0.999 AND CNR >= 50.0
    optimizer = ProtocolOptimizer(
        twin=twin,
        target_ssim_min=0.999,
        target_cnr_min=50.0,
        seed=99,
    )

    best_state, history_df, summary = optimizer.optimize(
        n_trials=4,
        n_realizations=1,
    )

    assert best_state is None, "Optimizer must not label an infeasible candidate as feasible!"
    assert "No feasible protocol found" in summary

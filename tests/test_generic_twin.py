"""Modality-independent digital twin, authoritative feasibility function and optimizer behaviour."""

import numpy as np

from config import NO_FEASIBLE_MESSAGE
from digital_twin.twin import DigitalTwin, evaluate_protocol, is_protocol_feasible
from modalities import get_modality, modality_names
from simulation.lesion import insert_lesion
from simulation.phantom import generate_shepp_logan


def _twin(name, target=True):
    base = generate_shepp_logan(64)
    m = get_modality(name)
    if target:
        p, l, b, meta = insert_lesion(base, radius=5.0, contrast=0.15)
        return DigitalTwin(m, p, l, b, meta)
    return DigitalTwin(m, base)


def test_evaluate_protocol_has_the_same_structure_for_every_modality():
    for name in modality_names():
        m = get_modality(name)
        res = evaluate_protocol(_twin(name), m.reference_protocol(), {"min_ssim": 0.1, "min_cnr": 0.01})
        for k in ("ssim", "psnr", "rmse", "cnr", "relative_exposure", "cost", "feasible"):
            assert k in res
        assert res["cost"] == 1.0 and res["cnr"] is not None and np.isfinite(res["cnr"])
        assert all(np.isfinite(v) for v in res.values() if isinstance(v, float))


def test_cnr_is_none_not_zero_without_a_target():
    for name in modality_names():
        res = evaluate_protocol(_twin(name, target=False), get_modality(name).reference_protocol(), {"min_ssim": 0.1})
        assert res["cnr"] is None


def test_is_protocol_feasible_is_the_single_authority():
    assert is_protocol_feasible({"ssim": 0.9}, {"min_ssim": 0.88})
    assert not is_protocol_feasible({"ssim": 0.87}, {"min_ssim": 0.88})
    assert is_protocol_feasible({"ssim": 0.9, "cnr": None}, {"min_ssim": 0.88, "min_cnr": 3.0})        # no target -> CNR skipped
    assert is_protocol_feasible({"ssim": 0.9, "cnr": float("nan")}, {"min_ssim": 0.88, "min_cnr": 3.0})
    assert not is_protocol_feasible({"ssim": 0.9, "cnr": 2.0}, {"min_ssim": 0.88, "min_cnr": 3.0})
    assert not is_protocol_feasible({"ssim": 0.9, "psnr": 20.0}, {"min_ssim": 0.88, "min_psnr": 25.0})
    assert is_protocol_feasible({"ssim": 0.9, "psnr": 30.0}, {"min_ssim": 0.88, "min_psnr": 25.0})
    assert not is_protocol_feasible({"ssim": float("nan")}, {"min_ssim": 0.0})


def test_optimizer_reports_no_feasible_protocol_for_every_modality():
    from optimization.protocol_optimizer import ProtocolOptimizer

    for name in modality_names():
        opt = ProtocolOptimizer(_twin(name), target_ssim_min=0.999, target_cnr_min=99.0, seed=3)
        best, hist, msg = opt.optimize(n_trials=3)
        assert best is None and NO_FEASIBLE_MESSAGE in msg and len(hist) == 3


def test_optimizer_best_state_is_feasible_and_cheapest_found():
    from optimization.protocol_optimizer import ProtocolOptimizer

    for name in modality_names():
        opt = ProtocolOptimizer(_twin(name), target_ssim_min=0.3, target_cnr_min=0.1, seed=4)
        best, hist, msg = opt.optimize(n_trials=6)
        if best is not None:
            assert best.is_feasible and best.ssim >= 0.3
            assert best.normalized_acquisition_cost == hist[hist["is_feasible"]]["normalized_acquisition_cost"].min()

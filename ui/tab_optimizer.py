"""Tab: Protocol Optimizer — Optuna TPE over the modality's search space."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from config import NO_FEASIBLE_MESSAGE
from digital_twin.controller import DigitalTwinController
from optimization.protocol_optimizer import ProtocolOptimizer
from ui.common import fmt, metric_grid, scope_box
from ui.compat import stretch


def render(ctx) -> None:
    m = ctx.modality
    st.subheader(f"Intelligent Acquisition Protocol Optimization ({m.name})")
    scope_box(m)
    st.markdown("Bayesian optimization (TPE) searches this modality's protocol space for the lowest-cost protocol that satisfies the "
                "sidebar criteria. A result is reported only if the digital twin confirms it.")
    c1, c2, c3 = st.columns(3)
    mode = c1.radio("Mode", ["Quick (15 trials, 1 realization)", "Research (40 trials, 3 realizations)"], key="opt_mode")
    trials, reals = (15, 1) if mode.startswith("Quick") else (40, 3)
    include_alt = c2.checkbox("Include SART in the search space (CT / Other only)", value=False, key="opt_sart",
                              disabled=m.name not in ("CT", "Other"), help="Iterative reconstruction is slower.")
    c3.caption(f"{trials} trials × {reals} noise realization(s) per candidate.")
    store = f"opt_{m.name}"
    if st.button("🚀 RUN PROTOCOL OPTIMIZER", type="primary", key="opt_go"):
        bar, txt = st.progress(0.0), st.empty()
        opt = ProtocolOptimizer(ctx.twin, ctx.min_ssim, ctx.eff_min_cnr, seed=ctx.seed, target_psnr_min=ctx.min_psnr)

        def cb(i, n, best):
            bar.progress(i / n)
            txt.text(f"Trial {i}/{n} | " + (f"best feasible so far: {m.describe_protocol(best.protocol)} "
                                            f"(cost {best.normalized_acquisition_cost:.3f})" if best else "searching for a feasible protocol…"))

        try:
            with st.spinner("Running optimization loop…"):
                best, hist, msg = opt.optimize(n_trials=trials, n_realizations=reals, include_sart=include_alt, progress_callback=cb)
            st.session_state[store] = (best, hist, msg, opt.controller.reference_state)
        except ImportError as exc:
            st.error(str(exc))
    if store in st.session_state:
        best, hist, msg, ref = st.session_state[store]
        st.markdown("---")
        (st.success if best is not None else st.error)(f"**Outcome:** {msg}")
        ctrl = DigitalTwinController(ctx.twin, ctx.min_ssim, ctx.eff_min_cnr, ctx.min_psnr)
        ctrl.reference_state = ref
        naive_p = min(m.training_grid(), key=lambda p: abs(m.acquisition_cost(p).cost - 0.1))
        naive, _, _ = ctx.twin.evaluate_protocol(naive_p, ctx.min_ssim, ctx.eff_min_cnr, target_psnr_min=ctx.min_psnr)
        st.dataframe(ctrl.compare_protocols(best, naive), hide_index=True, **stretch(st.dataframe))
        if best is not None:
            red = 100.0 * (1.0 - best.normalized_acquisition_cost / ref.normalized_acquisition_cost)
            metric_grid([(m.cost_labels["reduction"], f"{red:.1f}%", f"-{red:.1f}% vs reference"),
                         ("Optimal protocol cost", f"{best.normalized_acquisition_cost:.3f}",
                          f"{best.normalized_acquisition_cost - 1.0:+.3f} vs reference"),
                         ("SSIM / CNR", f"{fmt(best.ssim, '.3f')} / {fmt(best.cnr, '.2f')}")], cols=3)
        else:
            st.warning(NO_FEASIBLE_MESSAGE)
        if len(hist):
            h = hist.copy()
            h["status"] = h["is_feasible"].map({True: "feasible", False: "infeasible"})
            fig = px.scatter(h, x="normalized_acquisition_cost", y="ssim", color="status", hover_data=[c for c in h.columns if c in
                             [s.name for s in m.protocol_space()]], title="All trials: normalized cost vs SSIM (feasible = meets all criteria)",
                             color_discrete_map={"feasible": "#16A34A", "infeasible": "#DC2626"},
                             labels={"normalized_acquisition_cost": "Normalized acquisition cost", "ssim": "SSIM"})
            fig.add_hline(y=ctx.min_ssim, line_dash="dot", annotation_text="SSIM criterion")
            st.plotly_chart(fig, **stretch(st.plotly_chart))
            feas = h[h["is_feasible"]].sort_values("normalized_acquisition_cost")
            if len(feas):
                pareto, best_ssim = [], -1.0
                for _, r in feas.iterrows():          # cost-ascending frontier of best SSIM
                    if r["ssim"] > best_ssim:
                        pareto.append(r)
                        best_ssim = r["ssim"]
                st.caption("Pareto frontier of feasible trials (lowest cost for each SSIM gain):")
                st.dataframe(__import__("pandas").DataFrame(pareto)[["normalized_acquisition_cost", "ssim", "psnr"]], **stretch(st.dataframe))
            with st.expander("Trial history"):
                st.dataframe(hist, **stretch(st.dataframe))

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = "......................."
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from drift_module import (
    ConstrainedSTMConfig,
    Constraints,
    sample_beta_trajectories,
    sample_tilde_eta_and_theta,
    build_phi_from_beta,
)
from generate_synthetic_data import (
    build_covariates_synthetic,
    make_seed_r,
    ensure_min_len,
    mh_total_target,
)

# Frozen paper-corpus settings (synthetic_generation_v2.py main() defaults)
V, K_MH, K_BG, T, DOCS_PER_TIME = 900, 4, 12, 36, 50
DOC_LEN_MU, DOC_LEN_MIN = 50.0, 5
A_BG, A_MH, TAU_MH = 0.03, 1.0, 0.20
SIGMA_ETA, SIGMA_BETA, Q_BETA = 1.0, 0.30, 1.00
SEED_WORDS_PER_TOPIC = 5
GEN_SEED = 0

K = K_MH + K_BG
D = T * DOCS_PER_TIME


def purity_score(true_labels, pred_labels) -> float:
    """Identical to run_synthetic.py:purity_score (clustering purity)."""
    cm = contingency_matrix(true_labels, pred_labels)
    return float(np.sum(np.max(cm, axis=0)) / np.sum(cm))


def replay_generator():
    """Replay synthetic_generation_v2.generate()'s RNG stream in exact order.

    Order is load-bearing: is_mh_doc -> time offsets -> covariates -> N_d ->
    beta -> theta. Any deviation yields a different beta.
    """
    rng = np.random.default_rng(GEN_SEED)
    t_d = np.repeat(np.arange(T, dtype=int), DOCS_PER_TIME)

    seeds_per_mh = {
        k: list(range(k * SEED_WORDS_PER_TOPIC, (k + 1) * SEED_WORDS_PER_TOPIC))
        for k in range(K_MH)
    }
    tau = np.zeros(K, dtype=float)
    tau[:K_MH] = TAU_MH
    r = make_seed_r(K=K, V=V, mh_topics=list(range(K_MH)), seeds_per_mh=seeds_per_mh)

    # (1) minority-document activation
    u = t_d / (T - 1 + 1e-12)
    mh_rate = np.clip(mh_total_target(u) / max(A_MH, 1e-6), 0.02, 0.50)
    is_mh = rng.random(D) < mh_rate
    a = np.where(is_mh, A_MH, A_BG).astype(float)

    # (2) timestamp offsets -- consumed but unused downstream
    month_starts = pd.date_range(start="2021-01-01", periods=T, freq="MS")
    for t in range(T):
        start = month_starts[t]
        end = month_starts[t + 1] if t < T - 1 else start + pd.DateOffset(months=1)
        rng.integers(0, max(int((end - start).total_seconds()), 1), size=DOCS_PER_TIME)

    # (3) covariates (also draws votes / reply / heart)
    x, _meta = build_covariates_synthetic(D, t_d, T, rng, center=0.65)

    # (4) document lengths
    N_d = ensure_min_len(rng.poisson(lam=DOC_LEN_MU, size=D), DOC_LEN_MIN)

    # (5) topic-word trajectories
    cfg = ConstrainedSTMConfig(
        K=K, V=V, T=T, P=x.shape[1],
        Sigma=np.eye(K - 1) * SIGMA_ETA,
        Sigma_beta=np.eye(V) * SIGMA_BETA,
        Q_beta=np.eye(V) * Q_BETA,
    )
    beta = sample_beta_trajectories(cfg, rng)
    cons = Constraints(a=a, tau=tau, r=r)
    phi = build_phi_from_beta(beta, cons)

    # (6) document proportions -- B is deterministic, consumes no RNG
    B = np.zeros((x.shape[1], K - 1), dtype=float)
    for k in range(K - 1):
        B[0, k] = -0.30 if k < K_MH else 0.0
    B[3, 0] += 3.5   # U-shape
    B[2, 1] += 1.2   # seasonal
    B[1, 2] += 1.8   # rising
    B[4, 3] += 2.0   # bell
    _tilde_eta, theta = sample_tilde_eta_and_theta(
        x, B, cfg, cons, rng, list(range(K_MH))
    )

    return dict(theta=theta, phi=phi, N_d=N_d, is_mh=is_mh, t_d=t_d)


def verify(rep, csv_path):
    df = pd.read_csv(csv_path)
    csv_len = df["text"].str.split().str.len().to_numpy()
    csv_theta = df[[f"true_theta_{i}" for i in range(K_MH)]].to_numpy()
    checks = {
        "n_rows": bool(len(df) == D),
        "is_mh_doc": bool((rep["is_mh"] == df["is_mh_doc"].to_numpy()).all()),
        "doc_lengths": bool((rep["N_d"] == csv_len).all()),
        "true_theta": bool(np.allclose(rep["theta"][:, :K_MH], csv_theta)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Replay does not match frozen CSV: {checks}")
    return checks


def topic_distinctness(phi):
    ov, tv = [], []
    for t in range(T):
        for i in range(K_MH):
            for j in range(i + 1, K_MH):
                A, Bm = phi[i, t], phi[j, t]
                ov.append(float(np.minimum(A, Bm).sum()))
                tv.append(float(0.5 * np.abs(A - Bm).sum()))
    return dict(
        overlap_coefficient_mean=float(np.mean(ov)),
        overlap_coefficient_median=float(np.median(ov)),
        total_variation_mean=float(np.mean(tv)),
        total_variation_median=float(np.median(tv)),
    )


def evidence_stats(theta, N_d, idx):
    Tm, Nm = theta[idx, :K_MH], N_d[idx]
    srt = np.sort(Tm, axis=1)[:, ::-1]
    exp_label = Tm.max(axis=1) * Nm
    gap = (srt[:, 0] - srt[:, 1]) * Nm
    return dict(
        mh_mass_mean=float(Tm.sum(1).mean()),
        mh_mass_median=float(np.median(Tm.sum(1))),
        mh_mass_min=float(Tm.sum(1).min()),
        argmax_share_of_mh_mass_mean=float((Tm.max(1) / Tm.sum(1)).mean()),
        doc_length_mean=float(Nm.mean()),
        exp_tokens_any_mh_mean=float((Tm.sum(1) * Nm).mean()),
        exp_tokens_label_topic_median=float(np.median(exp_label)),
        exp_token_gap_top1_top2_median=float(np.median(gap)),
        frac_docs_gap_lt_3_tokens=float((gap < 3).mean()),
        frac_docs_gap_lt_5_tokens=float((gap < 5).mean()),
    )


def oracle_ceiling(theta, N_d, idx, labels, n_rep, seed):
    """Oracle knows true theta and the generating topic of every token.

    Labels by argmax minority-topic token count. Strictly more information than
    any word-based method has. Not a proven Bayes bound: a rule exploiting the
    class prior could in principle do better.
    """
    rng = np.random.default_rng(seed)
    K_all = theta.shape[1]
    pur, nmis, acc = [], [], []
    for _ in range(n_rep):
        pred = np.empty(len(idx), dtype=int)
        for i, d in enumerate(idx):
            z = rng.choice(K_all, size=int(N_d[d]), p=theta[d])
            cnt = np.bincount(z, minlength=K_all)[:K_MH]
            pred[i] = int(cnt.argmax()) if cnt.sum() > 0 else int(rng.integers(K_MH))
        pur.append(purity_score(labels, pred))
        nmis.append(float(normalized_mutual_info_score(labels, pred)))
        acc.append(float((pred == labels).mean()))
    return dict(
        purity_mean=float(np.mean(pur)), purity_std=float(np.std(pur)),
        nmi_mean=float(np.mean(nmis)), nmi_std=float(np.std(nmis)),
        raw_accuracy_mean=float(np.mean(acc)),
        n_repetitions=int(n_rep),
    )


def random_floor(labels, n_rep, seed):
    """Random assignment over K_MH clusters, same metric definitions."""
    rng = np.random.default_rng(seed)
    pur, nmis = [], []
    for _ in range(n_rep):
        pred = rng.integers(0, K_MH, size=len(labels))
        pur.append(purity_score(labels, pred))
        nmis.append(float(normalized_mutual_info_score(labels, pred)))
    return dict(
        purity_mean=float(np.mean(pur)), purity_std=float(np.std(pur)),
        nmi_mean=float(np.mean(nmis)), nmi_std=float(np.std(nmis)),
        n_repetitions=int(n_rep),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="frozen synthetic_data_fixedd.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_rep", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--drift_purity", type=float, default=0.670)
    ap.add_argument("--drift_nmi", type=float, default=0.296)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rep = replay_generator()
    checks = verify(rep, args.csv)

    theta, phi, N_d, is_mh = rep["theta"], rep["phi"], rep["N_d"], rep["is_mh"]
    idx = np.where(is_mh)[0]
    labels = theta[idx, :K_MH].argmax(axis=1)

    counts = np.bincount(labels, minlength=K_MH)
    majority = float(counts.max() / counts.sum())

    ceiling = oracle_ceiling(theta, N_d, idx, labels, args.n_rep, args.seed)
    floor = random_floor(labels, args.n_rep, args.seed + 1)

    def closed(v, lo, hi):
        return float((v - lo) / (hi - lo))

    res = {
        "replay_verification": checks,
        "n_scored_documents": int(len(idx)),
        "metric_definitions": {
            "scored_subset": "documents with is_mh_doc == True",
            "true_label": "argmax over true_theta_0..3",
            "purity": "clustering purity via contingency matrix (run_synthetic.py)",
            "nmi": "sklearn normalized_mutual_info_score",
        },
        "topic_distinctness": topic_distinctness(phi),
        "class_counts": counts.tolist(),
        "class_proportions": np.round(counts / counts.sum(), 4).tolist(),
        "majority_class_purity_baseline": majority,
        "evidence": evidence_stats(theta, N_d, idx),
        "random_guess_floor": floor,
        "oracle_reference_ceiling": ceiling,
        "drift_reported": {"purity": args.drift_purity, "nmi": args.drift_nmi},
        "fraction_of_range_closed_vs_random_floor": {
            "purity": round(closed(args.drift_purity, floor["purity_mean"],
                                   ceiling["purity_mean"]), 4),
            "nmi": round(closed(args.drift_nmi, floor["nmi_mean"],
                                ceiling["nmi_mean"]), 4),
        },
        "caveat": (
            "The oracle labels by argmax token count, approximately the MLE "
            "under a flat prior. It is a reference point, not a proven Bayes "
            "bound; a rule exploiting the class prior could in principle exceed it."
        ),
    }

    (out / "absolute_performance_context.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(out / "paper_corpus_ground_truth.npz", phi=phi, theta=theta)

    print(json.dumps(res, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
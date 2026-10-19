from __future__ import annotations
import argparse
from datetime import timedelta
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from drift_module import (
    ConstrainedSTMConfig,
    Constraints,
    sample_beta_trajectories,
    build_phi_from_beta,
    sample_tilde_eta_and_theta,
    generate_documents,
)

# # # #This control how many MH docs per month
def mh_total_target(u: np.ndarray) -> np.ndarray:
    # gentle seasonal variation around 12.5%
    return 0.125 + 0.025 * np.sin(2 * np.pi * u)   # ~10%..15%


def build_covariates_synthetic(D, t_d, T, rng, center=0.65, sigma=0.12):
    """
    P=5 covariates:
      x0 = 1 (intercept)
      x1 = u (linear)
      x2 = sin(2πu) (seasonal)
      x3 = (u-center)^2 (quadratic; + => U-shape, - => bell)
      x4 = exp(-0.5 * ((u-center)/sigma)^2) (bell)
    """
    #here we force the 4 MH topics to have different shapes
    u = t_d / (T - 1 + 1e-12)
    quad = (u - center) ** 2
    seasonal = np.sin(2 * np.pi * u)
    bell = np.exp(-0.5 * ((u - center) / sigma) ** 2)

    votes = rng.poisson(lam=3.0, size=D).astype(int)
    reply = (rng.random(D) < 0.15)
    heart = (rng.random(D) < 0.05)

    x = np.column_stack([np.ones(D), u, seasonal, quad, bell]).astype(float)
    # x= np.column_stack([np.log1p(votes),
    #     reply,
    #     heart,
    #     u,
    #     np.sin(2 * np.pi * u),
    #     (u - 0.5) ** 2]).astype(float)
    
    meta = {"votes": votes, "reply": reply, "heart": heart}
    return x, meta

def make_seed_r(K: int, V: int, mh_topics: list[int], seeds_per_mh: dict[int, list[int]]) -> np.ndarray:
    # main_module uses seed support via (r[k] > 0), not r magnitudes.
    # we keep positive mass only on MH seed words.
    r = np.zeros((K, V), dtype=float)
    for k in mh_topics:
        idx = np.array(seeds_per_mh[k], dtype=int)
        if idx.size > 0:
            r[k, idx] = 1.0 / idx.size
    return r


def suppress_seeds_in_background_beta(beta: np.ndarray, mh_seed_ids: list[int], K_MH: int, penalty: float = 1.0) -> np.ndarray:
    beta = beta.copy()
    seed_ids = np.array(mh_seed_ids, dtype=int)
    beta[K_MH:, :, seed_ids] -= penalty
    return beta

def ensure_min_len(N_d, min_len):
    N_d = N_d.astype(int)
    N_d[N_d < min_len] = min_len
    return N_d


def generate(
    out_csv: str,
    out_params_json: str,
    seed: int = 0,
    V: int = 900,
    K_MH: int = 4,
    K_BG: int = 12,
    T: int = 36,
    docs_per_time: int = 50,
    doc_len_mu: float = 50.0,
    doc_len_min: int = 5,
    a_bg: float = 0.03,       
    a_mh: float = 1.0,    
    tau_mh: float = 0.20,
    Sigma_eta_scale: float = 1.0,
    Sigma_beta: float = 0.30,
    Q_beta: float = 1.00,
    bg_seed_penalty: float = 0.0,
):
    rng = np.random.default_rng(seed)

    K = K_MH + K_BG
    mh_topics = list(range(K_MH))

    seed_words_per_topic = 5
    needed_seed_words = K_MH * seed_words_per_topic
    if needed_seed_words > V:
        raise ValueError(
            f"Need at least V >= {needed_seed_words} for {K_MH} MH topics "
            f"with {seed_words_per_topic} seed words/topic, got V={V}."
        )

    seeds_per_mh = {
        k: list(range(k * seed_words_per_topic, (k + 1) * seed_words_per_topic))
        for k in mh_topics
    }
    mh_seed_ids = sorted({wid for ids in seeds_per_mh.values() for wid in ids})

############################################################
    # constraints (tau, r)
    tau = np.zeros(K, dtype=float)
    tau[mh_topics] = tau_mh
    r = make_seed_r(K=K, V=V, mh_topics=mh_topics, seeds_per_mh=seeds_per_mh)
    D = T * docs_per_time
    t_d = np.repeat(np.arange(T, dtype=int), docs_per_time)
    u = t_d / (T - 1 + 1e-12)
    target = mh_total_target(u)  # if you keep this function
    mh_doc_rate_t = np.clip(target / max(a_mh, 1e-6), 0.02, 0.50)

    is_mh_doc = rng.random(D) < mh_doc_rate_t
    a = np.where(is_mh_doc, a_mh, a_bg).astype(float)

    month_starts = pd.date_range(start="2021-01-01", periods=T, freq="MS")
    time_iso = []
    for t in range(T):
        start = month_starts[t]
        end = month_starts[t + 1] if t < T - 1 else start + pd.DateOffset(months=1)
        seconds = int((end - start).total_seconds())
        offsets = rng.integers(0, max(seconds, 1), size=docs_per_time)
        for off in offsets:
            time_iso.append((start + timedelta(seconds=int(off))).isoformat())

    x, meta = build_covariates_synthetic(D, t_d, T, rng, center=0.65)
    P = x.shape[1]

    B = np.zeros((P, K - 1), dtype=float)
    for k in range(K - 1):
        B[0, k] = -0.30 if k in mh_topics else 0.0

    mh0, mh1, mh2, mh3 = 0, 1, 2, 3
    B[3, mh0] += +3.5    # U-shape   (quadratic):  logit swing ≈ 3.5×0.42 = 1.47 → 5.2× peak/trough
    B[2, mh1] += +1.2    # seasonal:  logit swing ≈ 1.2×2.0  = 2.40 → 12× peak/trough
    B[1, mh2] += +1.8    # linear:    logit swing ≈ 1.8×1.0  = 1.80 → 5.5× peak/trough
    B[4, mh3] += +2.0    # bell:      logit swing ≈ 2.0×1.0  = 2.00 → 6.6× peak/trough

    # bg_cols = [k for k in range(K - 1) if k not in mh_topics]
    # B[1:, bg_cols] += rng.normal(scale=0.08, size=(P - 1, len(bg_cols)))
    Sigma = np.eye(K - 1, dtype=float) * Sigma_eta_scale

    # u = t_d / (T - 1 + 1e-12)
    # target = mh_total_target(u)             
    # mh_doc_rate_t = np.clip(target / max(a_mh, 1e-6), 0.02, 0.50) #Then we convert it into a probability a doc is MH at time t

    # is_mh_doc = rng.random(D) < mh_doc_rate_t  
    # a = np.where(is_mh_doc, a_mh, a_bg).astype(float)#a_mh and a_bg control how strong constraints are per doc

    cons = Constraints(a=a, tau=tau, r=r)

    # config
    cfg = ConstrainedSTMConfig(
        K=K,
        V=V,
        T=T,
        P=P,
        Sigma=Sigma,
        Sigma_beta=np.eye(V) * float(Sigma_beta),
        Q_beta=np.eye(V) * float(Q_beta),
    )

    # doc lengths
    N_d = ensure_min_len(rng.poisson(lam=doc_len_mu, size=D), doc_len_min)
    beta = sample_beta_trajectories(cfg, rng)
    if bg_seed_penalty > 0:
        beta = suppress_seeds_in_background_beta(beta, mh_seed_ids, K_MH=K_MH, penalty=bg_seed_penalty)

    phi = build_phi_from_beta(beta, cons)
    tilde_eta, theta_true = sample_tilde_eta_and_theta(x, B, cfg, cons, rng, mh_topics)
    z_docs, w_docs = generate_documents(theta_true, phi, t_d, N_d, rng)

    mh_mass = theta_true[:, :K_MH].sum(axis=1)
    print(f"Avg MH mass:        {float(mh_mass.mean()):.4f}")
    print(f"MH mass min/max:    {float(mh_mass.min()):.4f} / {float(mh_mass.max()):.4f}")
    #print(f"Mean target m(u):   {float(target.mean()):.4f}")
    # print(f"Mean mh_doc_rate:   {float(mh_doc_rate_t.mean()):.4f}")
    # print(f"Mean is_mh_doc:     {float(is_mh_doc.mean()):.4f}")
    print(f"a_mh/a_bg ratio:    {a_mh/max(a_bg, 1e-9):.1f}×")
    #     # ── Ablation-validity sanity checks ────────────────────────────────────
    # # These catch the B-degeneracy problem before the ablation study runs.
    # from scipy.special import softmax as sp_softmax
    # u_arr = np.arange(T) / (T - 1 + 1e-12)
    # c_d_vals = []
    # for u in u_arr:
    #     xv = np.array([1.0, u, np.sin(2*np.pi*u), (u-0.65)**2,
    #                    np.exp(-0.5*((u-0.65)/0.12)**2)])
    #     mu = B.T @ xv
    #     tt = sp_softmax(np.append(mu, 0.0))
    #     c_d_vals.append(tt[:K_MH].sum())

    # c_d_max = max(c_d_vals)
    # degenerate_bins = sum(1 for c in c_d_vals if c > 0.90)

    # print(f"\n── Ablation-validity checks ─────────────────────────────")
    # print(f"  max c_d across time bins:  {c_d_max:.4f}  (limit: 0.90)")
    # print(f"  degenerate bins (c_d>0.90): {degenerate_bins}/T  (limit: 0)")

    # if c_d_max > 0.90:
    #     import warnings
    #     warnings.warn(
    #         f"\n\nABLATION VALIDITY WARNING\n"
    #         f"  max c_d = {c_d_max:.4f} at some time bins.\n"
    #         f"  {degenerate_bins}/{T} bins have c_d > 0.90.\n"
    #         f"  This causes the MH rescaling denominator (1-c_d) to approach zero,\n"
    #         f"  producing numerical blow-up in background topic weights.\n"
    #         f"  More critically: near-one-hot theta at these bins encodes temporal\n"
    #         f"  information directly in word content, making the temporal covariate\n"
    #         f"  redundant in inference — the no_prev_time ablation will be flat.\n"
    #         f"  FIX: reduce B magnitudes until max c_d < 0.60.\n",
    #         stacklevel=2,
    #     )

    vocab_tokens = np.array([f"word_{i}" for i in range(V)], dtype=object)
    texts = [" ".join(vocab_tokens[w]) for w in w_docs]
    seed_set = set(mh_seed_ids)
    has_seed = np.array([np.any(np.isin(w, list(seed_set))) for w in w_docs], dtype=bool)
    rows = []
    for i in range(D):
        row = {
            "doc_id": f"doc_{int(t_d[i])}_{i % docs_per_time}",
            "text": texts[i],
            "timestamp_bin": int(t_d[i]),
            "time_iso": time_iso[i],
            "votes": int(meta["votes"][i]),
            "reply": bool(meta["reply"][i]),
            "heart": bool(meta["heart"][i]),
            "channel": "synthetic_channel",
            "is_mh_doc": bool(is_mh_doc[i]),
            "has_seed": bool(has_seed[i]),
            "a_d": float(a[i]),
        }
        for k in range(K_MH):
            row[f"true_theta_{k}"] = float(theta_true[i, k])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)


    theta = theta_true  # (D, K)
    trend = np.zeros((T, K))
    for t in range(T):
        idx = np.where(t_d == t)[0]
        trend[t] = theta[idx].mean(axis=0)

    x_axis = month_starts

    fig, ax1 = plt.subplots(figsize=(14, 7))

    mh_names = ["Anxiety (U)", "Mental (Seasonal)", "Depression (Rising)", "Suicide (Bell)"]
    
    for k in range(K_MH):
        name_k = mh_names[k] if k < len(mh_names) else f"MH_{k}"
        ax1.plot(
            x_axis,
            trend[:, k],
            linewidth=3,
            # label=f"True MH {k}: {mh_names[k]}"
            label=f"True MH {k}: {name_k}"
        )

    ax1.set_ylabel("Mean MH topic prevalence", fontsize=13)
    ax1.set_ylim(0, 0.20)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    bg_sum = trend[:, K_MH:].sum(axis=1)
    ax2.plot(
        x_axis,
        bg_sum,
        linestyle="--",
        linewidth=3,
        alpha=0.6,
        label="True Background total"
    )
    ax2.set_ylabel("Mean background prevalence", fontsize=13, rotation=270, labelpad=18)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper right")
    plt.savefig("synthetic_true_trends.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


    params = {
        "seed": seed,
        "V": V,
        "K_MH": K_MH,
        "K_BG": K_BG,
        "K": K,
        "T": T,
        "docs_per_time": docs_per_time,
        "doc_len_mu": doc_len_mu,
        "doc_len_min": doc_len_min,
        "a_bg": a_bg,
        "a_mh": a_mh,
        "tau_mh": tau_mh,
        "Sigma_eta_scale": Sigma_eta_scale,
        "Sigma_beta_scalar": Sigma_beta,
        "Q_beta_scalar": Q_beta,
        "bg_seed_penalty": bg_seed_penalty,
        "mh_topics": mh_topics,
        "seeds_per_mh": seeds_per_mh,
        "B": B.tolist(),
        "Sigma": Sigma.tolist(),
        "tau": tau.tolist(),
        "r": r.tolist(),
    }
    with open(out_params_json, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"Saved: {out_csv}  (#docs={len(df)}, K={K}, T={T}, V={V})")
    print(f"Saved: {out_params_json}")
    #print(f"is_mh_doc: {is_mh_doc.mean():.3f} | has_seed: {has_seed.mean():.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=str, default="synthetic_data_fixedd.csv")
    ap.add_argument("--out_params_json", type=str, default="synthetic_params_stmm.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--V", type=int, default=900)
    ap.add_argument("--K_MH", type=int, default=4)
    ap.add_argument("--K_BG", type=int, default=12)
    ap.add_argument("--T", type=int, default=36)
    ap.add_argument("--docs_per_time", type=int, default=50)
    ap.add_argument("--doc_len_mu", type=float, default=50.0)
    ap.add_argument("--doc_len_min", type=int, default=5)
    ap.add_argument("--a_bg", type=float, default=0.03)
    ap.add_argument("--a_mh", type=float, default=1.0)
    ap.add_argument("--tau_mh", type=float, default=0.2)
    ap.add_argument("--Sigma_eta_scale", type=float, default=1.0)
    ap.add_argument("--Sigma_beta", type=float, default=0.30)
    ap.add_argument("--Q_beta", type=float, default=1.0)
    ap.add_argument("--bg_seed_penalty", type=float, default=0.0)
    args = ap.parse_args()

    generate(
        out_csv=args.out_csv,
        out_params_json=args.out_params_json,
        seed=args.seed,
        V=args.V,
        K_MH=args.K_MH,
        K_BG=args.K_BG,
        T=args.T,
        docs_per_time=args.docs_per_time,
        doc_len_mu=args.doc_len_mu,
        doc_len_min=args.doc_len_min,
        a_bg=args.a_bg,
        a_mh=args.a_mh,
        tau_mh=args.tau_mh,
        Sigma_eta_scale=args.Sigma_eta_scale,
        Sigma_beta=args.Sigma_beta,
        Q_beta=args.Q_beta,
        bg_seed_penalty=args.bg_seed_penalty,
    )

if __name__ == "__main__":
    main()

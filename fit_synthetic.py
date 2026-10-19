"""Fit DRIFT to the synthetic benchmark"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from drift_model import Constraints, VIConfig, fit_constrained_stm_vi, estimate_theta_docs_mc, to_shared_seed_mask


def run_synthetic_experiment(csv_path="synthetic_data.csv", params_path="synthetic_params.json", seed=0, log_path="training_log.csv"):
    df = pd.read_csv(csv_path)
    with open(params_path, "r") as f:
        params = json.load(f)
    V = int(params["V"])
    vocab = {f"word_{i}": i for i in range(V)}
    print(f"Vocab Size: {len(vocab)}")
    texts = df["text"].astype(str).tolist()
    all_tokens = [t.split() for t in texts]

    w_docs = []
    for tokens in all_tokens:
        ids = [vocab[w] for w in tokens if w in vocab]
        w_docs.append(np.asarray(ids, dtype=int))

    # --- time ---
    time_col = "timestamp_bin" if "timestamp_bin" in df.columns else "timestamp"
    t_d = df[time_col].astype(int).to_numpy()
    T = int(t_d.max()) + 1


    K = int(params["K"]) 
    mh_topics = [0, 1, 2, 3]

    # --- covariates: features of time ---
    u = t_d / (T - 1 + 1e-12)
    quad = (u - 0.65)**2
    seasonal = np.sin(2*np.pi*u)
    bell = np.exp(-0.5*((u-0.65)/0.12)**2)

    x = np.column_stack([np.ones(len(u)), u, seasonal, quad, bell])
    P = x.shape[1]
 


    a = np.where(df["has_seed"], 1.0, 0.03)                     
    r = to_shared_seed_mask(np.array(params["r"])
    B = np.zeros((x.shape[1], K - 1))                              
    Sigma = np.eye(K - 1)
    tau = np.array(params["tau"], dtype=float)
    cons = Constraints(a=a, tau=tau, r=r)


    print("Training on synthetic data...")
    vi_cfg = VIConfig(
        n_iter=40, 
        n_mc_elogtheta=5,
        n_mc_grad_m=5,
        lr_m=0.05,
        lr_beta=0.05,
        seed=seed
    )
    fit = fit_constrained_stm_vi(
        w_docs=w_docs, t_d=t_d, x=x, cons=cons,
        B=B, Sigma=Sigma, K=K, T=T, V=len(vocab),
        Sigma_beta=0.3, Q_beta=1.0, vi_cfg=vi_cfg, log_path=log_path)
    

    print("K,T,V:", K, T, V)
    print("x:", x.shape, "B:", B.shape, "Sigma:", Sigma.shape)
    print("a:", a.shape, "t_d:", t_d.shape, "num docs:", len(w_docs))

    assert B.shape[0] == x.shape[1]
    assert B.shape[1] == K-1
    assert Sigma.shape == (K-1, K-1)
    assert len(a) == len(w_docs) == len(t_d)

    
    return fit, vocab, df, cons


def purity_score(true_labels, pred_labels):
    cm = contingency_matrix(true_labels, pred_labels)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


def row_normalize(M, eps=1e-12):
    M = np.asarray(M, float)
    s = M.sum(axis=1, keepdims=True)
    return M / np.maximum(s, eps)


def compute_all_metrics(theta, df, mh_topics=[0,1,2,3]):
    y_pred_k = np.argmax(theta, axis=1)

    has_is_mh = "is_mh_doc" in df.columns

    # If no gating labels exist, binary MH metrics are undefined.
    if has_is_mh:
        is_mh = df["is_mh_doc"].astype(int).to_numpy()
        nmi_full = normalized_mutual_info_score(is_mh, y_pred_k)
        pur_full = purity_score(is_mh, y_pred_k)

        mh_set = set(mh_topics)
        y_pred_mh_bin = np.array([1 if k in mh_set else 0 for k in y_pred_k], dtype=int)
        nmi_mh_bin = normalized_mutual_info_score(is_mh, y_pred_mh_bin)
        pur_mh_bin = purity_score(is_mh, y_pred_mh_bin)

        idx = df["is_mh_doc"].astype(bool).to_numpy()
    else:
        nmi_full = np.nan
        pur_full = np.nan
        nmi_mh_bin = np.nan
        pur_mh_bin = np.nan
        idx = np.ones(len(df), dtype=bool)  # evaluate MH-4 on all docs

    true_mh_label = df[[f"true_theta_{i}" for i in range(4)]].to_numpy().argmax(axis=1)
    pred_mh_label = theta[:, mh_topics].argmax(axis=1)

    if idx.sum() > 0:
        nmi_mh_4 = normalized_mutual_info_score(true_mh_label[idx], pred_mh_label[idx])
        pur_mh_4 = purity_score(true_mh_label[idx], pred_mh_label[idx])
    else:
        nmi_mh_4, pur_mh_4 = np.nan, np.nan

    return {
        "NMI_full(binary_vs_K)": nmi_full,
        "Purity_full(binary_vs_K)": pur_full,
        "NMI_MH_binary": nmi_mh_bin,
        "Purity_MH_binary": pur_mh_bin,
        "NMI_MH_4class": nmi_mh_4,
        "Purity_MH_4class": pur_mh_4,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="synthetic_data.csv")
    ap.add_argument("--params", default="synthetic_params.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mh_topics = [0, 1, 2, 3]
    fit, vocab, df_true, cons = run_synthetic_experiment(csv_path=args.csv, params_path=args.params, seed=args.seed)
    theta_est = row_normalize(estimate_theta_docs_mc(fit["m"], fit["s"], cons, mh_topics, n_mc=50, seed=args.seed))
    m = compute_all_metrics(theta_est, df_true, mh_topics=mh_topics)
    print(f"NMI: {m['NMI_MH_4class']:.3f}  Purity: {m['Purity_MH_4class']:.3f}")

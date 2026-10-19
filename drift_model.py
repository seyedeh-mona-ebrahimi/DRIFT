from __future__ import annotations

import os, json, re, glob
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from scipy.optimize import minimize
import torch
import csv


# def initialize_beta_partitioned(beta, tau, r, mh_topics,
#                                   boost=1.0, suppress=-0.5):
#     for k in mh_topics:
#         own_seeds   = np.where(r[k] > 0)[0]
#         other_seeds = np.where(
#             (r[mh_topics, :].sum(axis=0) > 0) & (r[k] == 0)
#         )[0]
#         if len(own_seeds) > 0:
#             beta[k, :, own_seeds]   += boost
#         if len(other_seeds) > 0 and suppress != 0.0:
#             beta[k, :, other_seeds] += suppress
#     return beta


def diagnose_topic_diversity(phi, mh_topics):
    from itertools import combinations
    results = {}
    cosines = []

    for k1, k2 in combinations(mh_topics, 2):
        num = (phi[k1] * phi[k2]).sum(axis=-1)
        den = np.linalg.norm(phi[k1], axis=-1) * np.linalg.norm(phi[k2], axis=-1)
        cos = float((num / np.clip(den, 1e-12, None)).mean())
        cosines.append(cos)
        results[f"cos({k1},{k2})"] = round(cos, 4)

    results["mean_cosine"] = round(float(np.mean(cosines)), 4) if cosines else 0.0

    entropies = []
    for k in mh_topics:
        h = -np.sum(phi[k] * np.log(phi[k] + 1e-30), axis=-1)
        h_norm = h / np.log(phi.shape[-1])
        entropies.append(h_norm.mean())
        results[f"entropy({k})"] = round(float(h_norm.mean()), 3)

    results["mean_entropy"] = round(float(np.mean(entropies)), 3) if entropies else 0.0
    return results



def rw1_log_prior_beta(beta: np.ndarray, Sigma_beta: float, Q_beta: float) -> float:

    beta = np.asarray(beta, dtype=float)
    K, T, V = beta.shape

    sigma2 = float(Sigma_beta)
    q2 = float(Q_beta)

    # log N(x|0, sigma2 I): -0.5*(V*log(2pi*sigma2) + ||x||^2/sigma2)
    # log N(x_t|x_{t-1}, q2 I): -0.5*(V*log(2pi*q2) + ||x_t - x_{t-1}||^2/q2)
    lp = 0.0
    for k in range(K):
        b0 = beta[k, 0]
        lp += -0.5 * (V * np.log(2.0 * np.pi * sigma2) + np.sum(b0 * b0) / sigma2)
        for t in range(1, T):
            diff = beta[k, t] - beta[k, t - 1]
            lp += -0.5 * (V * np.log(2.0 * np.pi * q2) + np.sum(diff * diff) / q2)
    return float(lp)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True)




# ---------- Document-level prevalence latent: tilde_eta and theta ----------
def append_zero_last_dim(tilde_eta: np.ndarray) -> np.ndarray:
    """
    tilde_eta: shape (..., K-1)
    returns eta: shape (..., K) where last component is 0
    """
    tilde_eta = np.asarray(tilde_eta)
    zeros = np.zeros((*tilde_eta.shape[:-1], 1), dtype=tilde_eta.dtype)
    return np.concatenate([tilde_eta, zeros], axis=-1)


def phi_from_beta_new(
    beta_kt: np.ndarray,
    tau_k: float,
    seed_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    tilde_phi = softmax(beta_kt, axis=-1)  # (V,)

    if seed_indices.size == 0 or tau_k <= 0:
        # no constraint active
        phi = np.maximum(tilde_phi, 1e-30)
        return phi, tilde_phi

    S_tilde = float(np.sum(tilde_phi[seed_indices]))
    S_tilde = max(S_tilde, 1e-12)
    target = tau_k + (1.0 - tau_k) * S_tilde
    f_seed = target / S_tilde
    f_non  = 1.0 - tau_k
    phi = tilde_phi * f_non
    phi[seed_indices] = tilde_phi[seed_indices] * f_seed
    return phi, tilde_phi

# ---------- Small sampling helpers ----------
def mvn_sample(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from multivariate normal N(mean, cov).
    mean: (d,)
    cov:  (d,d)
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    return rng.multivariate_normal(mean=mean, cov=cov)


def categorical_sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    """
    probs: (K,) sums to 1
    returns an integer in {0,...,K-1}
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))

# ---------- Model configuration ----------
@dataclass
class ConstrainedSTMConfig:
    K: int        # number of topics
    V: int          # vocabulary size
    T: int          
    P: int          # covariate dimension for prevalence mu(x)
    Sigma: np.ndarray       # (K-1, K-1) covariance for tilde_eta
    Sigma_beta: np.ndarray  
    Q_beta: np.ndarray      # (V, V) innovation covariance for beta random walk

@dataclass
class Constraints:
    a: np.ndarray
    tau: np.ndarray
    r: np.ndarray


# ---------- Prevalence mean function mu(x) ----------

def mu_from_x(x: np.ndarray, B: np.ndarray) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    B = np.asarray(B, dtype=float)
    return x @ B  # (K-1,)

# ---------- Sample beta trajectories ----------
def sample_beta_trajectories(cfg: ConstrainedSTMConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Sample beta_{k,t} for all topics and times using:
      beta_{k,1} ~ N(0, Sigma_beta)
      beta_{k,t} | beta_{k,t-1} ~ N(beta_{k,t-1}, Q_beta)

    returns beta: shape (K, T, V)
    """
    K, T, V = cfg.K, cfg.T, cfg.V
    beta = np.zeros((K, T, V), dtype=float)

    zero_mean = np.zeros(V, dtype=float)

    for k in range(K):
        beta[k, 0] = mvn_sample(zero_mean, cfg.Sigma_beta, rng)
        for t in range(1, T):
            beta[k, t] = mvn_sample(beta[k, t - 1], cfg.Q_beta, rng)

    return beta


def build_phi_from_beta(beta: np.ndarray, cons: Constraints) -> np.ndarray:

    K, T, V = beta.shape
    phi = np.zeros((K, T, V), dtype=float)
    # for k in range(K):
    #     for t in range(T):
    #         phi[k, t], _ = phi_from_beta_and_seeds(beta[k, t], float(cons.tau[k]), cons.r[k])
     # --------------------
    for k in range(K):
        # Get indices where r > 0
        seed_indices = np.where(cons.r[k] > 0)[0]
        for t in range(T):
            phi[k, t], _ = phi_from_beta_new(beta[k, t], float(cons.tau[k]), seed_indices)
    # --------------------
    return phi

def torch_softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)


def phi_from_beta_new_torch(beta_kt: torch.Tensor, tau_k: float, seed_mask: torch.Tensor) -> torch.Tensor:
    """
    Implements the seed-mass rescaling construction.

    Inputs:
      beta_kt   : (V,) torch tensor (requires_grad=True)
      tau_k     : scalar in [0,1)
      seed_mask : (V,) boolean tensor, True for seed words in S_k

    Output:
      phi       : (V,) probability vector that sums to 1 (by construction)
                 and satisfies sum_{v in S_k} phi[v] >= tau_k
    """
    tilde_phi = torch.softmax(beta_kt, dim=-1)  # (V,)

    # If no constraint active, return plain softmax (with floor for log safety)
    if (tau_k <= 0.0) or (seed_mask is None) or (not bool(seed_mask.any())):
        return torch.clamp(tilde_phi, min=1e-30)

    S_tilde = tilde_phi[seed_mask].sum()
    S_tilde = torch.clamp(S_tilde, min=1e-12)

    target = tau_k + (1.0 - tau_k) * S_tilde
    f_seed = target / S_tilde
    f_non  = (1.0 - tau_k)

    phi = tilde_phi * f_non
    phi = torch.where(seed_mask, tilde_phi * f_seed, phi)

    # ONLY floor clip for log safety; DO NOT renormalize
    # phi = torch.clamp(phi, min=1e-30)
    #return phi
    return torch.clamp(phi, min=1e-30)

def update_beta_rw1_autograd(
    beta: np.ndarray,          # (K,T,V)
    counts: np.ndarray,        # (K,T,V)
    tau: np.ndarray,           # (K,)
    r: np.ndarray,       
    lr: float,
    Sigma_beta: float,         # scalar variance for t=0
    Q_beta: float,             # scalar RW variance
    #use_temporal_prior: bool = True,
    beta_prior_mode: str = "rw1",
) -> np.ndarray:
    K, T, V = beta.shape
    new_beta = beta.copy()

    inv_sigma2 = 1.0 / float(Sigma_beta)
    inv_q2 = 1.0 / float(Q_beta)

    for k in range(K):
        tau_k = float(tau[k])
        seed_mask = torch.tensor((r[k] > 0), dtype=torch.bool)

        # Optimize all time slices jointly for topic k
        B = torch.tensor(beta[k], dtype=torch.float32, requires_grad=True)   # (T,V)
        C = torch.tensor(counts[k], dtype=torch.float32)                     # (T,V)

        # Likelihood: sum_{t,v} counts[k,t,v] * log phi_{k,t,v}
        loglik = torch.tensor(0.0, dtype=torch.float32)
        for t in range(T):
            phi_t = phi_from_beta_new_torch(B[t], tau_k, seed_mask)
            loglik = loglik + torch.sum(C[t] * torch.log(phi_t))

        # RW1 prior in edge form:
        # -0.5/sigma2 * ||B[0]||^2 - 0.5/q2 * sum_{t=1}^{T-1} ||B[t]-B[t-1]||^2
        prior = torch.tensor(0.0, dtype=torch.float32)
        if beta_prior_mode == "rw1":
            prior = prior - 0.5 * inv_sigma2 * torch.sum(B[0] * B[0])
            if T > 1:
                dB = B[1:] - B[:-1] #forward edge equivalent to the sum over t=2 to T
                prior = prior - 0.5 * inv_q2 * torch.sum(dB * dB)

        elif beta_prior_mode == "iid":
            # simple non-temporal prior in which treats each time point independently
            prior = prior - 0.5 * inv_sigma2 * torch.sum(B * B)

        elif beta_prior_mode == "none":
            pass
        
        obj = loglik + prior
        obj.backward()

        with torch.no_grad():
            G = torch.clamp(B.grad, -5.0, 5.0)
            B_new = B + lr * G

        new_beta[k] = B_new.detach().cpu().numpy()

    return new_beta




# ---------- Sample document-level tilde_eta and theta ----------
def sample_tilde_eta_and_theta(
    x: np.ndarray,           # (D,P)
    B: np.ndarray,           # (P,K-1)
    cfg: ConstrainedSTMConfig,
    cons: Constraints,
    rng: np.random.Generator,
    mh_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    D = x.shape[0]
    K = cfg.K
    tilde_eta = np.zeros((D, K - 1), dtype=float)
    theta = np.zeros((D, K), dtype=float)
    for d in range(D):
        mu_d = mu_from_x(x[d], B)                 # (K-1,)
        tilde_eta[d] = mvn_sample(mu_d, cfg.Sigma, rng)
        scalar_a = float(cons.a[d])
        #a_d = cons.a[d]           # scalar or (K,) — do NOT cast to float
        #scalar_a = float(np.mean(a_d)) if np.ndim(a_d) > 0 else float(a_d)
        #theta[d] = theta_from_tilde_eta_per_topic_mh(tilde_eta[d], scalar_a, mh_indices)
        theta[d] = theta_from_tilde_eta_scalar_mh(tilde_eta[d], scalar_a, mh_indices)
        
    return tilde_eta, theta

# ---------- Generate tokens ----------
""""
z_dn = categorical_sample(theta[d]): Draws a topic.
w_dn = categorical_sample(phi[z_dn, td]): Draws a word 
                                    based on that topic and time."""
def generate_documents(
    theta: np.ndarray,   # (D,K)
    phi: np.ndarray,     # (K,T,V)
    t_d: np.ndarray,     # (D,) time indices in {0,...,T-1}
    N_d: np.ndarray,     # (D,) token counts
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    For each document d:
      for n=1..N_d:
        z_dn ~ Cat(theta_d)
        w_dn ~ Cat(phi_{z_dn, t_d})

    returns:
      z_docs: list length D, each array shape (N_d,)
      w_docs: list length D, each array shape (N_d,) with word ids in {0,...,V-1}
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    t_d = np.asarray(t_d, dtype=int)
    N_d = np.asarray(N_d, dtype=int)
    D, K = theta.shape
    _, T, V = phi.shape

    z_docs: List[np.ndarray] = []
    w_docs: List[np.ndarray] = []
    for d in range(D):
        td = int(t_d[d])
        if not (0 <= td < T):
            raise ValueError(f"t_d[{d}]={td} out of range 0..{T-1}")

        Nd = int(N_d[d])
        z = np.zeros(Nd, dtype=int)
        w = np.zeros(Nd, dtype=int)

        for n in range(Nd):
            z_dn = categorical_sample(theta[d], rng)          # topic index 0..K-1
            w_dn = categorical_sample(phi[z_dn, td], rng)     # word index 0..V-1
            z[n] = z_dn
            w[n] = w_dn
        z_docs.append(z)
        w_docs.append(w)
    return z_docs, w_docs


# ---------- Full generative sampler wrapper ----------
def sample_constrained_stm(
    cfg: ConstrainedSTMConfig,
    cons: Constraints,
    x: np.ndarray,          # (D,P)
    t_d: np.ndarray,        # (D,) in {0,...,T-1}
    N_d: np.ndarray,        # (D,)
    B: np.ndarray,          # (P,K-1)
    seed: int = 0,
    mh_indices: List[int] = None) -> Dict[str, object]:

    rng = np.random.default_rng(seed)
    beta = sample_beta_trajectories(cfg, rng)          # (K,T,V)
    phi = build_phi_from_beta(beta, cons)              # (K,T,V)
    # tilde_eta, theta = sample_tilde_eta_and_theta(x, B, cfg, cons, rng)  # (D,K-1), (D,K)
    tilde_eta, theta = sample_tilde_eta_and_theta(x, B, cfg, cons, rng, mh_indices)
    z_docs, w_docs = generate_documents(theta, phi, t_d, N_d, rng)
    return {
        "beta": beta,
        "phi": phi,
        "tilde_eta": tilde_eta,
        "theta": theta,
        "z_docs": z_docs,
        "w_docs": w_docs,
    }


def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return y if keepdims else np.squeeze(y, axis=axis)

def build_log_phi(beta: np.ndarray, tau: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    beta: (K,T,V)
    returns:
      log_phi: (K,T,V)
      phi:     (K,T,V)
      s:       (K,T,V) residual softmax(beta)
    """
    K, T, V = beta.shape
    phi = np.zeros((K, T, V), dtype=float)
    s = np.zeros((K, T, V), dtype=float)
    # for k in range(K):
    #     for t in range(T):
    #         phi[k, t], s[k, t] = phi_from_beta_and_seeds(beta[k, t], float(tau[k]), r[k])
    for k in range(K):
        # --------------------
        seed_indices = np.where(r[k] > 0)[0]
        for t in range(T):
            phi[k, t], s[k, t] = phi_from_beta_new(beta[k, t], float(tau[k]), seed_indices)
    log_phi = np.log(phi + 1e-30) 
    return log_phi, phi, s

#I samples eps, computes theta, takes log, and averages.
def expected_log_theta_mc(
    m_d: np.ndarray,          # (K-1,)
    s_d: np.ndarray,          # (K-1,)
    # a_d: np.ndarray,          # (K,)
    a_d_scalar: float,      
    mh_indices: List[int],    
    n_samples: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Monte Carlo: sample tilde_eta ~ N(m_d, diag(s_d^2)), compute log theta, average.
    Returns E[log theta] shape (K,)
    """
    K_minus_1 = m_d.shape[0]
    logs = []
    for _ in range(n_samples):
        eps = rng.normal(size=K_minus_1)
        tilde_eta = m_d + s_d * eps
        theta =theta_from_tilde_eta_scalar_mh(tilde_eta, a_d_scalar, mh_indices)
        # theta =theta_from_tilde_eta_per_topic_mh(tilde_eta, a_d_scalar, mh_indices)
        if np.isnan(theta).any():
            # fallback to uniform or safe value if NaN appears
            theta = np.ones_like(theta) / len(theta)
        logs.append(np.log(theta + 1e-30)) 
    return np.mean(np.stack(logs, axis=0), axis=0)


def update_gamma_for_document(
    w_d: np.ndarray,           # (N_d,) word ids 0..V-1
    t_d: int,
    elog_theta_d: np.ndarray,  # (K,)
    log_phi: np.ndarray        # (K,T,V)
) -> np.ndarray:
    """
    gamma_dn(k) ∝ exp( Elog theta_d(k) + log phi_{k,t_d,w_dn} )
    returns gamma_d: (N_d, K)
    """
    K = elog_theta_d.shape[0]
    N = w_d.shape[0]
    scores = np.zeros((N, K), dtype=float)
    for n in range(N):
        v = int(w_d[n])
        scores[n] = elog_theta_d + log_phi[:, t_d, v]
    lse = logsumexp(scores, axis=1, keepdims=True)
    scores = scores - lse
    gamma = np.exp(scores)
    return gamma, np.sum(lse)



def grad_loglik_wrt_tilde_eta_new(
    tilde_eta: np.ndarray,
    a_d: float,
    n_k: np.ndarray,
    mh_indices: List[int]
) -> np.ndarray:
    
    # 1. Forward Pass
    eta = append_zero_last_dim(tilde_eta)
    p = softmax(eta)
    c_mh = np.sum(p[mh_indices])

    # c_mh = np.minimum(c_mh, 0.999) 
    eps = 1e-6
    c_mh = np.clip(c_mh, eps, 1.0 - eps)

    # S = (1 - a * c_mh) / (1 - c_mh)
    denom = 1.0 - c_mh
    num = 1.0 - a_d * c_mh
    S = num / denom
    theta = np.zeros_like(p)
    theta[mh_indices] = a_d * p[mh_indices]
    
    non_mh_mask = np.ones(len(p), dtype=bool)
    non_mh_mask[mh_indices] = False
    theta[non_mh_mask] = p[non_mh_mask] * S
    
    dL_dp = np.zeros_like(p)
    dL_dp[non_mh_mask] = n_k[non_mh_mask] / (p[non_mh_mask] + 1e-12)
    dS_dpk = (1.0 - a_d) / (denom * denom)
    sum_n_non_mh = np.sum(n_k[non_mh_mask])
    term_interaction = (sum_n_non_mh / (S + 1e-12)) * dS_dpk
    dL_dp[mh_indices] = (n_k[mh_indices] / (theta[mh_indices] + 1e-12)) * a_d + term_interaction
    term1 = p * dL_dp
    term2 = p * np.sum(term1)
    grad_eta = term1 - term2
    
    return grad_eta[:-1]


def update_m_d_gradient_ascent(
    m_d: np.ndarray,
    s_d: np.ndarray,
    a_d_scalar: float,         
    n_k: np.ndarray,
    mu_d: np.ndarray,
    Sigma_inv: np.ndarray,
    lr: float = 0.05,
    n_mc: int = 5,
    rng: Optional[np.random.Generator] = None,
    mh_indices: List[int] = None
) -> np.ndarray:
    """
    Reparameterization gradient ascent on m_d.
    Now uses the NEW multiplicative gradient function.
    """
    if mh_indices is None:
        raise ValueError("mh_indices are not existand must be provided.")
    if rng is None:
        rng = np.random.default_rng(0)
    K_minus_1 = m_d.shape[0]
    K = n_k.shape[0] 
    grad = np.zeros_like(m_d)
    for _ in range(n_mc):
        eps = rng.normal(size=K_minus_1)
        tilde_eta = m_d + s_d * eps
        g_sample = grad_loglik_wrt_tilde_eta_new(
            tilde_eta, 
            a_d_scalar, 
            n_k, 
            mh_indices
        )
        grad += g_sample
    grad /= max(n_mc, 1)
    grad_prior = -Sigma_inv @ (m_d - mu_d)
    m_new = m_d + lr * (grad + grad_prior)
    return m_new


def compute_expected_word_counts(
    w_docs,
    t_d,
    gamma_docs,
    K: int,
    T: int,
    V: int
) -> np.ndarray:
    counts = np.zeros((K, T, V), dtype=np.float64)
    k_idx = np.arange(K)[:, None]  # (K,1)
    for d, w in enumerate(w_docs):
        if w.size == 0:
            continue
        td = int(t_d[d])
        ww = w.astype(np.int64)
        gamT = gamma_docs[d].T  # (K, N)
        # add into (K,V) slice using (k_idx, ww[None,:])
        np.add.at(counts[:, td, :], (k_idx, ww[None, :]), gamT)
    return counts


#Gradient-Based Variational Inference with the Reparameterization Trick.
def update_log_s_d_gradient_ascent(
    m_d: np.ndarray,          # (K-1,)
    log_s_d: np.ndarray,      # (K-1,) unconstrained
    # a_d: np.ndarray,          # (K,)
    a_d_scalar: float,
    n_k: np.ndarray,          # (K,)
    mu_d: np.ndarray,         # (K-1,)
    Sigma_inv: np.ndarray,    # (K-1,K-1)
    lr: float,
    n_mc: int,
    rng: np.random.Generator,
    mh_indices: List[int],
    min_s: float = 1e-4,
    max_s: float = 10.0,
) -> np.ndarray:
    """
    True VI update for s_d (diagonal std) via reparameterization gradients,
    INCLUDING:
      - likelihood term
      - expected Gaussian prior term
      - entropy term
    """
    s_d = np.exp(log_s_d)
    # s_d = np.clip(s_d, min_s, max_s)
    K_minus_1 = m_d.shape[0]
    Sigma_inv_diag = np.diag(Sigma_inv) 
    grad_s_lik = np.zeros(K_minus_1, dtype=float)
    for _ in range(n_mc):
        eps = rng.normal(size=K_minus_1)
        tilde_eta = m_d + s_d * eps
        g_eta = grad_loglik_wrt_tilde_eta_new(tilde_eta, a_d_scalar, n_k, mh_indices)  # (K-1,)
        grad_s_lik += g_eta * eps
    grad_s_lik /= max(n_mc, 1)
    grad_s_prior = -Sigma_inv_diag * s_d
    grad_s_ent = 1.0 / np.clip(s_d, min_s, None)
    grad_s_total = grad_s_lik + grad_s_prior + grad_s_ent
    grad_log_s = grad_s_total * s_d

    log_s_new = log_s_d + lr * grad_log_s
    s_new = np.clip(np.exp(log_s_new), min_s, max_s)
    return np.log(s_new)

@dataclass
class VIConfig:
    n_iter: int = 40
    n_mc_elogtheta: int = 5
    n_mc_grad_m: int = 5
    n_mc_grad_s: int = 5
    lr_m: float = 0.05
    lr_s: float = 0.05
    lr_beta: float = 0.05
    seed: int = 0
#################################################################################################
# Main VI function
#################################################################################################

def fit_constrained_stm_vi(
    w_docs: List[np.ndarray],   # list length D, each (N_d,)
    t_d: np.ndarray,            # (D,) time in 0..T-1
    x: np.ndarray,              # (D,P)
    # a: np.ndarray,              # (D,K)
    cons: Constraints,
    # tau: np.ndarray,            # (K,)
    # r: np.ndarray,              # (K,V)
    B: np.ndarray,              
    Sigma: np.ndarray,          
    K: int,
    T: int,
    V: int,
    Sigma_beta: float | np.ndarray,
    Q_beta: float | np.ndarray,
    vi_cfg: VIConfig,
    log_path: str = "training_log.csv",
    # use_temporal_prior=True, 
    beta_prior_mode: str = "rw1",
    mh_topics_fixed: Optional[List[int]] = None,
    ) -> Dict[str, object]:


    rng = np.random.default_rng(vi_cfg.seed)
    a = cons.a
    tau = cons.tau
    r = cons.r
    D = len(w_docs)
    K_minus_1 = K - 1
    Sigma_inv = np.linalg.inv(Sigma)
    # Initialize variational params
    m = rng.normal(scale=0.05, size=(D, K-1))
    log_s = np.log(np.ones((D, K_minus_1), dtype=float) * 0.7)
    beta = rng.normal(scale=0.01, size=(K, T, V))
    if mh_topics_fixed is not None:
        mh_topics = list(mh_topics_fixed)
    else:
        mh_topics = np.where(tau > 0)[0].tolist()
    elbo_traj = []

    gamma_docs = []
    for d, w in enumerate(w_docs):
        Nd = len(w)
        if Nd == 0:
            gamma_docs.append(np.zeros((0, K)))
        else:
            g = rng.dirichlet(alpha=np.ones(K), size=Nd)  # (Nd,K)
            gamma_docs.append(g)



    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iter", "ELBO", "A_Prior", "B_TopicAssign", "C_Likelihood", "D1_CatEntropy", "D2_GaussEntropy", "Objective", "LogP_Beta"])

    beta_init = beta.copy()
    phi_init = build_phi_from_beta(beta, cons)
    beta_history = [beta.copy()]
    phi_history = [phi_init.copy()]
    for it in range(1, vi_cfg.n_iter + 1):
        # Build log_phi from current beta
        log_phi, phi, s_resid = build_log_phi(beta, tau, r)
        # ----- E-step: update gamma and m -----
        for d in range(D):
            mu_d = mu_from_x(x[d], B)                       # (K-1,)
   
            scalar_a = float(cons.a[d])
            s_d = np.exp(log_s[d])
            elog_theta_d = expected_log_theta_mc(
                m[d], s_d, scalar_a, mh_topics, vi_cfg.n_mc_elogtheta, rng)
            # gamma update
            gamma_d, _ = update_gamma_for_document(w_docs[d], int(t_d[d]), elog_theta_d, log_phi)
            gamma_docs[d] = gamma_d
            n_k = gamma_d.sum(axis=0)            # (K,)

            m[d] = update_m_d_gradient_ascent(m_d=m[d],s_d=s_d,
            a_d_scalar=scalar_a,n_k=n_k,mu_d=mu_d,Sigma_inv=Sigma_inv,
            lr=vi_cfg.lr_m,n_mc=vi_cfg.n_mc_grad_m,rng=rng,mh_indices=mh_topics)
    
            log_s[d] = update_log_s_d_gradient_ascent(m_d=m[d],log_s_d=log_s[d],
            a_d_scalar=scalar_a,n_k=n_k,mu_d=mu_d,Sigma_inv=Sigma_inv,lr=vi_cfg.lr_s,
            n_mc=vi_cfg.n_mc_grad_s,rng=rng,mh_indices=mh_topics)

        # ----- M-step: update beta using expected counts (local parameters)-----
        counts = compute_expected_word_counts(w_docs, t_d, gamma_docs, K, T, V)
        beta = update_beta_rw1_autograd(beta, counts, tau, r, vi_cfg.lr_beta, Sigma_beta, Q_beta,beta_prior_mode=beta_prior_mode)
        phi_now = build_phi_from_beta(beta, cons)
        beta_history.append(beta.copy())
        phi_history.append(phi_now.copy())


        # # ----- Update Global Parameters -----
        if it % 1 == 0:
            B, Sigma = update_global_params_M_step(x, m, log_s, B)
            Sigma_inv = np.linalg.inv(Sigma)
        if it % 1 == 0:
            parts = compute_elbo_parts_ABCD(
                w_docs=w_docs,
                t_d=t_d,
                x=x,
                cons=cons,
                B=B,
                Sigma=Sigma,
                beta=beta,
                gamma_docs=gamma_docs,
                m=m,
                log_s=log_s,
                mh_topics=mh_topics,
                Sigma_beta=float(Sigma_beta),
                Q_beta=float(Q_beta),
                n_mc_theta=vi_cfg.n_mc_elogtheta,
                rng=rng)
            # n_inc = K * V * (T - 1)
            #print("logp(beta)/inc =", parts["log_p_beta"] / n_inc)
            elbo_traj.append(parts["ELBO"])
            print(                                                                                                      
                f"iter {it:03d} | ELBO={parts['ELBO']:.2f} | "
                f"A={parts['A_prevalence_prior']:.2f} | "
                f"B={parts['B_topic_assignment']:.2f} | "
                f"C={parts['C_word_likelihood']:.2f} | "
                f"D1={parts['D1_cat_entropy']:.2f} | "
                f"D2={parts['D2_gauss_entropy']:.2f} | "  
                f"OBJ={parts['objective_ELBO_plus_logpbeta']:.2f} | logp(beta)={parts['log_p_beta']:.2f}")
            


            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    it,
                    parts['ELBO'],
                    parts['A_prevalence_prior'],
                    parts['B_topic_assignment'],
                    parts['C_word_likelihood'],
                    parts['D1_cat_entropy'],
                    parts['D2_gauss_entropy'],
                    parts['objective_ELBO_plus_logpbeta'],
                    parts['log_p_beta']])
    return {
        "m": m,
        "s": np.exp(log_s),
        "log_s": log_s,
        "beta": beta,
        "gamma_docs": gamma_docs,
        "phi": build_log_phi(beta, tau, r)[1],
        "elbo": np.array(elbo_traj),
        "B": B,         
        "Sigma": Sigma,
        "beta_init": beta_init,
        "phi_init": phi_init,
        "beta_history": beta_history,
        "phi_history": phi_history

    }


def theta_from_tilde_eta_scalar_mh(
    tilde_eta: np.ndarray,
    a_scalar: float,
    mh_indices: List[int],
) -> np.ndarray:
    eta = append_zero_last_dim(tilde_eta)
    tilde_theta = softmax(eta, axis=-1)
    if len(mh_indices) == 0:
        return tilde_theta
    c_mh = float(np.sum(tilde_theta[mh_indices]))
    c_mh_safe = min(c_mh, 1.0 - 1e-6)
    max_possible_a = (1.0 - 1e-6) / (c_mh_safe + 1e-12)
    a_eff = min(a_scalar, max_possible_a)
    theta = np.zeros_like(tilde_theta)
    theta[mh_indices] = a_eff * tilde_theta[mh_indices]
    non_mh_mask = np.ones(len(theta), dtype=bool)
    non_mh_mask[mh_indices] = False
    boosted_mass = a_eff * c_mh_safe
    scale_non_mh = (1.0 - boosted_mass) / (1.0 - c_mh_safe)
    theta[non_mh_mask] = tilde_theta[non_mh_mask] * scale_non_mh
    return theta


def estimate_theta_docs_mc(m, s, cons, mh_topics, n_mc=1, seed=0):
    rng = np.random.default_rng(seed)
    D, K_minus_1 = m.shape
    K = K_minus_1 + 1
    theta_hat = np.zeros((D, K), dtype=float)

    for d in range(D):
        a_d = float(cons.a[d])
        #scalar_a = float(cons.a[d])
        #a_d = cons.a[d]
        acc = np.zeros(K, dtype=float)
        for _ in range(max(n_mc, 1)):
            eps = rng.normal(size=K_minus_1)
            tilde_eta = m[d] + s[d] * eps
            acc += theta_from_tilde_eta_scalar_mh(tilde_eta, a_d, mh_topics)
            #acc += theta_from_tilde_eta_per_topic_mh(tilde_eta, a_d, mh_topics)
        theta_hat[d] = acc / max(n_mc, 1)

    return theta_hat
def to_shared_seed_mask(r_per_topic: np.ndarray, mh_topics: list[int]) -> np.ndarray:
    
    union_seed_ids = sorted({
        int(v) for k in mh_topics
        for v in np.where(r_per_topic[k] > 0)[0].tolist()
    })
    r_shared = np.zeros_like(r_per_topic)
    for k in mh_topics:
        r_shared[k, union_seed_ids] = 1.0
    return r_shared
def topic_prevalence_by_time(theta_hat: np.ndarray, t_d: np.ndarray, T: int) -> np.ndarray:
    K = theta_hat.shape[1]
    prev = np.zeros((T, K), dtype=float)
    for t in range(T):
        idx = np.where(t_d == t)[0]
        prev[t] = theta_hat[idx].mean(axis=0) if len(idx) else np.nan
    return prev

def update_global_params_M_step(x: np.ndarray, m: np.ndarray, log_s: np.ndarray, B_old: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates B (regression weights) and Sigma (covariance).
    Model: m_d ~ N(B^T x_d, Sigma)
    """
    D = x.shape[0]
    P = x.shape[1]
    # 1. Update B using Ridge Regression: B = (X^T X + lambda*I)^-1 X^T m
    lam = 1.0
    xtx = x.T @ x + lam * np.eye(P)
    xtm = x.T @ m
    B_new = np.linalg.solve(xtx, xtm)
    # 2. Update Sigma based on residuals
    # Sigma = (1/D) * sum( (m - Bx)(m - Bx)^T )
    residuals = m - (x @ B_new) # (D, K-1).  # Calculate (m - mu) for all docs
    Sigma_new = (residuals.T @ residuals) / D    # Matrix multiplication sums outer products  
    Sigma_new += np.diag(np.mean(np.exp(2*log_s), axis=0)) 
    # Add jitter for stability
    Sigma_new += np.eye(Sigma_new.shape[0]) * 0.001
    return B_new, Sigma_new

def term_A_eq_log_p_eta_given_x(
    m_d: np.ndarray,           # (K-1,)
    s_d: np.ndarray,           # (K-1,)
    mu_d: np.ndarray,          # (K-1,)
    Sigma_inv: np.ndarray,     # (K-1,K-1)
    log_det_Sigma: float
) -> float:
    """
    (A) E_q[ log N(tilde_eta_d | mu_d, Sigma) ]  with q=N(m_d, diag(s_d^2))

    E_q[log p] = -1/2 [ (K-1)log(2pi) + log|Sigma|
                       + (m-mu)^T Sigma^{-1} (m-mu)
                       + tr(Sigma^{-1} diag(s^2)) ]
    """
    m_d = np.asarray(m_d, float)
    s_d = np.asarray(s_d, float)
    mu_d = np.asarray(mu_d, float)
    d = m_d.size
    diff = (m_d - mu_d).reshape(-1, 1)
    # quad_mean = float(diff.T @ Sigma_inv @ diff)
    quad_mean = np.asarray(diff.T @ Sigma_inv @ diff).squeeze().item()
    quad_var = float(np.sum(np.diag(Sigma_inv) * (s_d ** 2)))
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det_Sigma + quad_mean + quad_var)


def term_B_eq_log_p_z_given_theta_mc(
    gamma_d: np.ndarray,      # (N_d,K)
    m_d: np.ndarray,          # (K-1,)
    s_d: np.ndarray,          # (K-1,)
    a_d_scalar: float,
    mh_topics: list[int],
    n_mc: int,
    rng: np.random.Generator
) -> float:
    """
    (B) sum_n sum_k gamma_dn(k) * E_q[ log theta_{d,k}(tilde_eta_d) ]

    where E_q[log theta] is approximated by MC via reparameterization.
    """
    elog_theta = expected_log_theta_mc(
        m_d=m_d,
        s_d=s_d,
        a_d_scalar=a_d_scalar,
        mh_indices=mh_topics,
        n_samples=n_mc,
        rng=rng
    )  # (K,)
    return float(np.sum(gamma_d * elog_theta[None, :]))

def term_C_eq_log_p_w_given_z_beta(
    w_d: np.ndarray,       # (N_d,)
    t_d: int,
    gamma_d: np.ndarray,   # (N_d,K)
    log_phi: np.ndarray    # (K,T,V)
) -> float:
    """
    (C) sum_n sum_k gamma_dn(k) * log phi_{k, t_d, w_dn}
    """
    total = 0.0
    for n, v in enumerate(w_d):
        total += float(np.dot(gamma_d[n], log_phi[:, t_d, int(v)]))
    return total


def term_D1_categorical_entropy_q_z(gamma_d: np.ndarray) -> float:
    """
    (D1) Sum_n H(Cat(gamma_dn))
    H = - sum_k gamma_dn(k) log gamma_dn(k)
    """
    gamma_d = np.asarray(gamma_d, float)
    return float(-np.sum(gamma_d * np.log(gamma_d + 1e-30)))


def term_D2_gaussian_entropy_q_eta(log_s_d: np.ndarray) -> float:
    """
    (D2) Gaussian entropy for q_d(tilde_eta_d)=N(m_d, diag(s_d^2))

    H(q_d) = 0.5 * sum_i log(2*pi*e*s_i^2)
           = 0.5 * sum_i [log(2*pi*e) + 2*log(s_i)]
    where log_s_d = log(s_d).
    """
    log_s_d = np.asarray(log_s_d, dtype=float)
    return 0.5 * float(np.sum(np.log(2.0 * np.pi * np.e) + 2.0 * log_s_d))


def compute_elbo_parts_ABCD(
    w_docs: list[np.ndarray],
    t_d: np.ndarray,
    x: np.ndarray,
    cons,                 # must have cons.a, cons.tau, cons.r
    B: np.ndarray,
    Sigma: np.ndarray,
    beta: np.ndarray,
    gamma_docs: list[np.ndarray],
    m: np.ndarray,
    log_s: np.ndarray,
    mh_topics: list[int],
    Sigma_beta: float,
    Q_beta: float,
    n_mc_theta: int,
    rng: np.random.Generator
) -> dict:
    """
    Computes ELBO decomposition parts:
      ELBO = (A) + (B) + (C) + (D1) + (D2)
      Objective = ELBO + log p(beta)
    We also return:
      neg_elbo = -ELBO 
    """
    D = len(w_docs)
    K, T, V = beta.shape
    # build phi(beta) deterministically
    log_phi, phi, _ = build_log_phi(beta, cons.tau, cons.r)
    # Sigma stuff
    Sigma_inv = np.linalg.inv(Sigma)
    sign, log_det_Sigma = np.linalg.slogdet(Sigma)
    if sign <= 0:
        Sigma = Sigma + 1e-6 * np.eye(Sigma.shape[0])
        Sigma_inv = np.linalg.inv(Sigma)
        sign, log_det_Sigma = np.linalg.slogdet(Sigma)

    A = Bterm = C = D1 = D2 = 0.0
    for d in range(D):
        w_d = w_docs[d]
        if w_d.size == 0:
            continue
        td = int(t_d[d])
        gamma_d = gamma_docs[d]
        m_d = m[d]
        s_d = np.exp(log_s[d])
        a_d = float(cons.a[d])
        #a_d = cons.a[d]
        mu_d = x[d] @ B
        A += term_A_eq_log_p_eta_given_x(m_d, s_d, mu_d, Sigma_inv, log_det_Sigma)

        Bterm += term_B_eq_log_p_z_given_theta_mc(gamma_d, m_d, s_d, a_d, mh_topics, n_mc_theta, rng)
        
        C += term_C_eq_log_p_w_given_z_beta(w_d, td, gamma_d, log_phi)

        D1 += term_D1_categorical_entropy_q_z(gamma_d)

        D2 += term_D2_gaussian_entropy_q_eta(log_s[d])


    ELBO = float(A + Bterm + C + D1 + D2)
    log_p_beta = rw1_log_prior_beta(beta, Sigma_beta=Sigma_beta, Q_beta=Q_beta)
    # n_inc = K * V * (T - 1)
    # print("logp(beta)/inc =", log_p_beta / n_inc)
    objective = ELBO + float(log_p_beta)

    return {
        "A_prevalence_prior": float(A),
        "B_topic_assignment": float(Bterm),
        "C_word_likelihood": float(C),
        "D1_cat_entropy": float(D1),
        "D2_gauss_entropy": float(D2),
        "ELBO": float(ELBO),
        "log_p_beta": float(log_p_beta),
        "objective_ELBO_plus_logpbeta": float(objective),
        "neg_ELBO_(KL_up_to_constant)": float(-ELBO),
    }





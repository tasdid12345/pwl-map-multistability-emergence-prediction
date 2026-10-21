"""Can a balanced, CV-tuned LightGBM on multi-initial-condition features detect
multistability in a 1D piecewise-linear (PWL) map better than logistic regression
on single-trajectory summaries?  Seeded numpy simulation, no download."""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, SEEDS = 42, (42, 43, 44)
N_SYS = 4000                     # simulated map parameter sets
N_IC_LAB, T_LAB = 200, 500       # ground truth: 200 initial conditions, 500 steps
N_IC_FEAT, T_FEAT = 16, 100      # improved model: 16 initial conditions, 100 steps
T_TRAJ = 500                     # baseline: one trajectory from x0 = 0.5
TOL = 1e-3                       # final states closer than this share an attractor
MIN_FRAC = 0.05                  # an attractor must hold >= 5% of initial conditions


def step(par, x):
    """One PWL map step: x <- clip(a*x - b*|x| + c, -2, 2), vectorised over systems."""
    a, b, c = par
    return np.clip(a * x - b * np.abs(x) + c, -2, 2)


def iterate(par, x, steps):
    """Iterate all initial conditions (rows = systems) for `steps` steps."""
    for _ in range(steps):
        x = step(par, x)
    return x


def trajectory(par, steps):
    """Single trajectory from x0 = 0.5 for every system, shape (N, steps)."""
    x, out = np.full((len(par[0]), 1), 0.5), []
    for _ in range(steps):
        x = step(par, x)
        out.append(x[:, 0])
    return np.column_stack(out)


def cluster_sizes(states):
    """Sizes of groups of sorted final states whose neighbours differ by <= TOL."""
    s = np.sort(states)
    cuts = np.where(np.diff(s) > TOL)[0] + 1
    return np.diff(np.concatenate(([0], cuts, [len(s)])))


def ic_feats(fin):
    """Multi-IC features: cluster counts, spread, end-state fractions per system."""
    rows = []
    for f in fin:
        cs = np.sort(cluster_sizes(f))[::-1]          # cluster sizes, largest first
        top = np.zeros(3)
        k = min(3, len(cs))
        top[:k] = cs[:k] / len(f)                     # top-3 end-state fractions
        rows.append([len(cs), np.sum(cs >= 2), f.std(), np.mean(f > 0), *top])
    return np.array(rows)


def fit_baseline(seed, Xtr, ytr, Xte):
    """Standardised logistic regression (default settings, no class weights)."""
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def fit_improved(seed, Xtr, ytr, Xte):
    """Balanced LightGBM, small 5-fold CV grid (train only), sigmoid calibration."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        LGBMClassifier(class_weight="balanced", random_state=seed, verbose=-1),
        {"learning_rate": [0.05, 0.1], "num_leaves": [15, 31], "n_estimators": [200, 400]},
        scoring="f1", cv=cv, n_jobs=1)
    grid.fit(Xtr, ytr)
    cal = CalibratedClassifierCV(grid.best_estimator_, method="sigmoid", cv=5)
    cal.fit(Xtr, ytr)
    return cal.predict_proba(Xte)[:, 1], grid.best_params_


def evaluate(yte, proba):
    """Main metric F1 (multistable class, threshold 0.5) plus ROC-AUC and Brier."""
    return {"f1": f1_score(yte, (proba >= 0.5).astype(int)),
            "roc_auc": roc_auc_score(yte, proba),
            "brier": brier_score_loss(yte, proba)}


def main():
    rng = np.random.default_rng(SEED)
    # Sample PWL map parameters a, b, c uniformly
    P = np.column_stack([rng.uniform(-1.5, 1.5, N_SYS),
                         rng.uniform(0.0, 1.5, N_SYS),
                         rng.uniform(-1.0, 1.0, N_SYS)])
    par = (P[:, 0:1], P[:, 1:2], P[:, 2:3])

    print("simulating ground-truth labels (200 ICs, 500 steps) ...")
    fin_lab = iterate(par, rng.uniform(-2, 2, (N_SYS, N_IC_LAB)), T_LAB)
    y = np.array([np.sum(cluster_sizes(f) >= MIN_FRAC * N_IC_LAB) >= 2 for f in fin_lab], int)
    print(f"multistable rate: {y.mean():.3f}")
    if y.min() == y.max():
        print("degenerate labels, cannot evaluate"); sys.exit(1)

    print("building baseline features (one trajectory from x0=0.5) ...")
    tail = trajectory(par, T_TRAJ)[:, -200:]
    X_base = np.column_stack([P, tail.mean(1), tail.std(1), tail.min(1), tail.max(1)])

    print("building improved features (16 ICs, 100 steps) ...")
    x16 = np.tile(np.linspace(-2, 2, N_IC_FEAT), (N_SYS, 1))
    X_imp = np.hstack([P, ic_feats(iterate(par, x16, T_FEAT))])

    # Single stratified 75/25 split shared by both models
    idx_tr, idx_te = train_test_split(np.arange(N_SYS), test_size=0.25,
                                      stratify=y, random_state=SEED)
    ytr, yte = y[idx_tr], y[idx_te]

    rb, ri, best = [], [], []
    for seed in SEEDS:
        print(f"seed {seed}: baseline logistic regression ...")
        rb.append(evaluate(yte, fit_baseline(seed, X_base[idx_tr], ytr, X_base[idx_te])))
        print(f"seed {seed}: improved LightGBM (grid search + calibration) ...")
        pi, bp = fit_improved(seed, X_imp[idx_tr], ytr, X_imp[idx_te])
        best.append(bp)
        ri.append(evaluate(yte, pi))
        print(f"  best CV params: {bp}")

    def agg(rs, k):
        v = np.array([r[k] for r in rs])
        return float(v.mean()), float(v.std())

    bf, bf_sd = agg(rb, "f1")
    imf, imf_sd = agg(ri, "f1")
    metrics = {
        "baseline_score": bf, "improved_score": imf,
        "improvement": imf - bf,
        "baseline_score_std": bf_sd, "improved_score_std": imf_sd,
        "baseline_f1": bf, "improved_f1": imf,
        "baseline_roc_auc": agg(rb, "roc_auc")[0], "improved_roc_auc": agg(ri, "roc_auc")[0],
        "baseline_brier": agg(rb, "brier")[0], "improved_brier": agg(ri, "brier")[0],
        "multistable_rate": float(y.mean()),
    }
    vals = list(metrics.values())
    if not all(np.isfinite(v) for v in vals):
        print("non-finite metrics"); sys.exit(1)

    os.makedirs("figures", exist_ok=True) if False else None
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["baseline (LogReg)", "improved (LightGBM)"], [bf, imf],
           yerr=[bf_sd, imf_sd], capsize=6, color=["grey", "steelblue"])
    ax.set_ylabel("F1 (multistable class, test)")
    ax.set_title("Mean over seeds 42-44")
    fig.tight_layout()
    fig.savefig("figures/f1_comparison.png", dpi=120)

    results = {
        "question": ("Can a class-balanced, CV-tuned gradient-boosting model using multi-initial-"
                     "condition trajectory features detect multistability in a monoparametric PWL "
                     "map better than logistic regression on a single-trajectory summary?"),
        "method": ("Seeded simulation of x_{n+1}=clip(a x - b|x| + c, -2, 2); label from 200 ICs "
                   "over 500 steps (>=2 attractors each holding >=5% of ICs); stratified 75/25 "
                   "split; baseline logistic regression vs LightGBM with balanced weights, 5-fold "
                   "CV grid and sigmoid calibration; seeds 42-44."),
        "dataset": {
            "name": "Simulated monoparametric piecewise-linear map (seeded, no download)",
            "source_url": "https://numpy.org",
            "n_rows": N_SYS,
            "n_train": int(len(idx_tr)),
            "n_test": int(len(idx_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "F1 on multistable class (test set)",
        "higher_is_better": True,
        "baseline": ("Standardised logistic regression on the map parameters plus mean, std, min, "
                     "max of one trajectory from x0=0.5 (500 steps, last 200 kept)."),
        "improved": ("LightGBM with balanced class weights, small 5-fold CV grid search on the "
                     "training set, sigmoid calibration; features are the parameters plus cluster "
                     "counts, spread and end-state fractions from 16 initial conditions over 100 steps."),
        "metrics": metrics,
        "notes": ("Simulated data, not the paper's system; results depend on the sampling ranges "
                  "and the clustering tolerance. The baseline is a standard stand-in, not the "
                  "paper's method. Threshold 0.5 fixed for F1; no tuning on the test set. "
                  "Logistic regression is deterministic so its std reflects no seed variation. "
                  "The 16-IC/100-step features may not have converged for slow transients, which "
                  "limits the improved model. No downloads, so no HTTP retry logic is needed."),
    }
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"baseline F1 {bf:.4f} +/- {bf_sd:.4f} | improved F1 {imf:.4f} +/- {imf_sd:.4f}")
    print("wrote results.json")


if __name__ == "__main__":
    main()

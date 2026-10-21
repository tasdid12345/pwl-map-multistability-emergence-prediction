# Multistability detection in a monoparametric PWL map: boosted multi-IC features vs. logistic regression

Predicting when a parameterised dynamical system acquires multiple coexisting attractors is hard because a single trajectory can look deceptively simple. The paper "Predicting the emergence of multistability in a Monoparametric PWL system" (2022) studies exactly this question for piecewise-linear maps. This repository is a small, self-contained experiment inspired by that setting; it does not reproduce the paper's method or data.

**Question.** Can a class-balanced, CV-tuned gradient-boosting model that uses multi-initial-condition trajectory features detect multistability better than a logistic regression on a single-trajectory summary?

**Dataset.** A seeded numpy simulation of the 1D piecewise-linear map x_{n+1} = clip(a x - b|x| + c, -2, 2) with parameters sampled uniformly (about 4,000 systems). No external download. This is a simulated dataset, not the paper's own data.

**Baseline vs. improved.**
- Baseline: standardised logistic regression on the parameters plus mean, std, min and max of one trajectory from x0 = 0.5. This is a standard stand-in, not the paper's method.
- Improved: LightGBM with balanced class weights, a small 5-fold CV grid search, and sigmoid calibration, using the parameters plus cluster counts, spread and end-state fractions from 16 initial conditions over 100 steps.

Both models use the same stratified 75/25 split, the same test set, and the same F1 metric on the multistable class, with ROC-AUC and Brier score as secondary metrics.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (and `figures/f1_comparison.png`).

## Results (latest run)

On a seeded simulated monoparametric piecewise-linear map (1,000 test points), a standard logistic regression on single-trajectory summaries reached F1 0.843 for the multistable class, while my class-balanced LightGBM on multi-initial-condition features reached 0.983.

| Metric | Value |
|---|---|
| baseline_score | 0.8433 |
| improved_score | 0.9831 |
| improvement | 0.1398 |
| baseline_score_std | 0 |
| improved_score_std | 0.00198 |
| baseline_f1 | 0.8433 |
| improved_f1 | 0.9831 |
| baseline_roc_auc | 0.9522 |
| improved_roc_auc | 0.9995 |
| baseline_brier | 0.07751 |
| improved_brier | 0.00828 |
| multistable_rate | 0.3563 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulated data, not the paper's system; results depend on the sampling ranges and the clustering tolerance. The baseline is a standard stand-in, not the paper's method. Threshold 0.5 fixed for F1; no tuning on the test set. Logistic regression is deterministic so its std reflects no seed variation. The 16-IC/100-step features may not have converged for slow transients, which limits the improved model. No downloads, so no HTTP retry logic is needed.

## Reference

This project is motivated by the following paper:

> JL Echenausía-Monroy, S Jafari (2022). *Predicting the emergence of multistability in a Monoparametric PWL system*. … Journal of Bifurcation. https://www.worldscientific.com/doi/abs/10.1142/S0218127422502066

Paper author: Prof. Héctor Gilardi, Universidad Panamericana (hgilardi@up.edu.mx)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

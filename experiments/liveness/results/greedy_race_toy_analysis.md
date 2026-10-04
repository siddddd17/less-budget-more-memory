
## toy: n=163 candidates {'mm': 113, '_scaled_dot_product_flash_attention_for_cpu': 16, 'mean': 33, 'nll_loss_forward': 1}
all-banned (aggressive min-cut, greedy start): peak 102.23 MB, R 0
greedy (greedy_lazy): 66 drops, converged=True, peak 58.86 MB, R 5.767 GF, oracle calls 1127 (3 full sweeps), 730s wall (2 worker(s)); +light 58.33 MB R 5.767; +all 58.33 MB R 7.571

| budget | method | peak MB | vs stock | recompute GF | oracle calls | plan s |
|---|---|---|---|---|---|---|
| 0.05 | stock | 148.00 | +0.0% | 14.019 | 0 | 0.0 |
| 0.05 | ours (LivenessAwareSolver) | 95.21 | -35.7% | 15.165 | 5 | 6.5 |
| 0.05 | stock + light remat | 81.38 | -45.0% | 14.019 | 0 | 8.9 |
| 0.05 | stock + all-ops remat | 55.89 | -62.2% | 28.918 | 0 | 6.7 |
| 0.05 | ours + light remat | 65.26 | -55.9% | 15.165 | 5 | 12.6 |
| 0.05 | greedy converged (budget-free) | 58.86 | -60.2% | 5.767 | 1127 | 730.0 |
| 0.05 | greedy + light remat | 58.33 | -60.6% | 5.767 | 1127 | 733.2 |
| 0.05 | greedy matched to ours (95.21) | 95.12 | -35.7% | 0.902 | 596 | 420.5 |
| 0.05 | greedy matched to ours+light (65.26) | 64.12 | -56.7% | 4.870 | 860 | 609.1 |
| 0.10 | stock | 66.58 | +0.0% | 11.230 | 0 | 0.0 |
| 0.10 | ours (LivenessAwareSolver) | 59.40 | -10.8% | 11.671 | 6 | 8.9 |
| 0.10 | stock + light remat | 57.83 | -13.1% | 11.230 | 0 | 3.3 |
| 0.10 | stock + all-ops remat | 55.89 | -16.1% | 13.797 | 0 | 2.9 |
| 0.10 | ours + light remat | 57.83 | -13.1% | 11.671 | 6 | 11.5 |
| 0.10 | greedy converged (budget-free) | 58.86 | -11.6% | 5.767 | 1127 | 730.0 |
| 0.10 | greedy + light remat | 58.33 | -12.4% | 5.767 | 1127 | 733.2 |
| 0.10 | greedy matched to ours (59.40) | 59.40 | -10.8% | 5.499 | 932 | 662.4 |
| 0.10 | greedy matched to ours+light (57.83) | not reached (min 58.86) | | | | |
| 0.15 | stock | 61.17 | +0.0% | 9.720 | 0 | 0.1 |
| 0.15 | ours (LivenessAwareSolver) | 59.44 | -2.8% | 9.720 | 5 | 7.0 |
| 0.15 | stock + light remat | 59.59 | -2.6% | 9.720 | 0 | 3.1 |
| 0.15 | stock + all-ops remat | 58.55 | -4.3% | 10.303 | 0 | 2.5 |
| 0.15 | ours + light remat | 57.87 | -5.4% | 9.720 | 5 | 9.7 |
| 0.15 | greedy converged (budget-free) | 58.86 | -3.8% | 5.767 | 1127 | 730.0 |
| 0.15 | greedy + light remat | 58.33 | -4.6% | 5.767 | 1127 | 733.2 |
| 0.15 | greedy matched to ours (59.44) | 59.40 | -2.9% | 5.499 | 932 | 662.4 |
| 0.15 | greedy matched to ours+light (57.87) | not reached (min 58.86) | | | | |
| 0.20 | stock | 65.20 | +0.0% | 8.781 | 0 | 0.0 |
| 0.20 | ours (LivenessAwareSolver) | 59.40 | -8.9% | 8.781 | 5 | 7.6 |
| 0.20 | stock + light remat | 63.62 | -2.4% | 8.781 | 0 | 3.0 |
| 0.20 | stock + all-ops remat | 62.22 | -4.6% | 9.456 | 0 | 3.0 |
| 0.20 | ours + light remat | 57.83 | -11.3% | 8.781 | 5 | 11.5 |
| 0.20 | greedy converged (budget-free) | 58.86 | -9.7% | 5.767 | 1127 | 730.0 |
| 0.20 | greedy + light remat | 58.33 | -10.5% | 5.767 | 1127 | 733.2 |
| 0.20 | greedy matched to ours (59.40) | 59.40 | -8.9% | 5.499 | 932 | 662.4 |
| 0.20 | greedy matched to ours+light (57.83) | not reached (min 58.86) | | | | |
| 0.30 | stock | 70.83 | +0.0% | 6.919 | 0 | 0.0 |
| 0.30 | ours (LivenessAwareSolver) | 59.40 | -16.1% | 6.921 | 5 | 8.1 |
| 0.30 | stock + light remat | 69.25 | -2.2% | 6.919 | 0 | 3.4 |
| 0.30 | stock + all-ops remat | 67.85 | -4.2% | 7.820 | 0 | 4.0 |
| 0.30 | ours + light remat | 57.83 | -18.4% | 6.921 | 5 | 11.0 |
| 0.30 | greedy converged (budget-free) | 58.86 | -16.9% | 5.767 | 1127 | 730.0 |
| 0.30 | greedy + light remat | 58.33 | -17.6% | 5.767 | 1127 | 733.2 |
| 0.30 | greedy matched to ours (59.40) | 59.40 | -16.1% | 5.499 | 932 | 662.4 |
| 0.30 | greedy matched to ours+light (57.83) | not reached (min 58.86) | | | | |

### absolute targets (fraction of stock@0.05 peak = 148.00 MB): cheapest recompute GF that reaches it

| method | 80% (118.4 MB) | 60% (88.8 MB) | 40% (59.2 MB) | min peak reached |
|---|---|---|---|---|
| stock (budget sweep 0..1) | 0.00 GF (b=0.68, 102.2) | 3.20 GF (b=0.50, 86.3) | no | 59.40 MB @ 10.33 GF |
| ours (5 budgets) | 6.92 GF (b=0.30, 59.4) | 6.92 GF (b=0.30, 59.4) | no | 59.40 MB @ 11.67 GF |
| stock+light | 6.92 GF (b=0.30, 69.3) | 6.92 GF (b=0.30, 69.3) | 11.23 GF (b=0.10, 57.8) | 57.83 MB @ 11.23 GF |
| stock+all | 7.82 GF (b=0.30, 67.8) | 7.82 GF (b=0.30, 67.8) | 10.30 GF (b=0.15, 58.5) | 55.89 MB @ 28.92 GF |
| ours+light | 6.92 GF (b=0.30, 57.8) | 6.92 GF (b=0.30, 57.8) | 6.92 GF (b=0.30, 57.8) | 57.83 MB @ 11.67 GF |
| greedy (trajectory) | 0.00 GF (step 0, 102.2) | 1.80 GF (step 42, 88.1) | 5.77 GF (step 66, 58.9) | 58.86 MB @ 5.77 GF |
| greedy + light (final) | 5.77 GF (final, 58.3) | 5.77 GF (final, 58.3) | 5.77 GF (final, 58.3) | 58.33 MB @ 5.77 GF |

stock sweep min peak 59.40 MB at b=0.12; all-banned 102.23 MB

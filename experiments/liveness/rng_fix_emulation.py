"""
rng_fix_emulation.py -- emulate the upstream fix for pytorch#190758 (draft PR #190759) locally.

Bug: under activation_memory_budget < 1, the knapsack can put banned RNG ops (dropout, rand_like, ...)
into `dont_ban`, so they are recomputed in the backward with fresh randomness and the gradients are wrong.
PR #190759's fix for 0 < budget < 1: exclude RNG ops from the knapsack's recompute candidates, so the
ban is final and their outputs are saved.

Emulation (v2, matches the actual PR #190759 diff for 0 < budget < 1):
  (0) should_ban_recomputation also bans `is_rng_op` nodes. This is OpTypes.is_random OR is_rng_op, so
      Inductor's prims.inductor_seeds (tagged nondeterministic_seeded) is banned under the
      aggressive min-cut options.
Then wrap solve_min_cut so that
  (a) RNG nodes are removed from the `banned_nodes` it returns (so they never become knapsack candidates), and
  (b) RNG nodes are removed from any `dont_ban` passed in (so they can never be unbanned).
Every caller goes through this wrapper: choose_saved_values_set, the stock solver, and our oracle.
Budget == 0 is not covered (it takes an early-return path we never use).

Usage: import rng_fix_emulation; rng_fix_emulation.install()
"""
import torch._functorch.partitioners as P
from torch.utils._ordered_set import OrderedSet

_installed = {"on": False}


def _is_rng(n) -> bool:
    try:
        if P.is_rng_op(n):
            return True
    except Exception:
        pass
    try:
        return P.get_default_op_list().is_random(n)
    except Exception:
        return False


def install() -> None:
    if _installed["on"]:
        return
    orig_is_random = P.OpTypes.is_random
    P.OpTypes.is_random = lambda self, n: orig_is_random(self, n) or P.is_rng_op(n)
    orig = P.solve_min_cut

    def solve_min_cut(joint_graph, node_info, min_cut_options, dont_ban=None):
        if dont_ban:
            dont_ban = OrderedSet(n for n in dont_ban if not _is_rng(n))
        saved, banned = orig(joint_graph, node_info, min_cut_options, dont_ban)
        return saved, OrderedSet(n for n in banned if not _is_rng(n))

    P.solve_min_cut = solve_min_cut
    _installed["on"] = True

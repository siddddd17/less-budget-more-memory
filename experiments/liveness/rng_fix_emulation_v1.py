"""
rng_fix_emulation_v1.py -- RECONSTRUCTION of the first version of the RNG-fix emulation (LOCAL ONLY).

v1 was used for results/gpu_eval_remat_bert_fixed.json (whose aot_eager rows are in the paper's Table 1 and
Section 6). The original file was later edited in place into v2 (rng_fix_emulation.py), so it is reconstructed here:
it is v2 minus the OpTypes.is_random patch, i.e. it only
  (a) removes RNG nodes from the `banned_nodes` returned by solve_min_cut (so they are never knapsack candidates), and
  (b) removes RNG nodes from any `dont_ban` passed in (so they can never be unbanned).
v1 is incomplete under Inductor (prims.inductor_seeds is never banned), which is why the paper uses v2 for every
Inductor row. Under aot_eager, check_rng_fix_v1_v2_equiv.py shows v1 and v2 choose identical saved sets.
"""
import torch._functorch.partitioners as P
from torch.utils._ordered_set import OrderedSet

from rng_fix_emulation import _is_rng

_installed = {"on": False}


def install() -> None:
    if _installed["on"]:
        return
    orig = P.solve_min_cut

    def solve_min_cut(joint_graph, node_info, min_cut_options, dont_ban=None):
        if dont_ban:
            dont_ban = OrderedSet(n for n in dont_ban if not _is_rng(n))
        saved, banned = orig(joint_graph, node_info, min_cut_options, dont_ban)
        return saved, OrderedSet(n for n in banned if not _is_rng(n))

    P.solve_min_cut = solve_min_cut
    _installed["on"] = True

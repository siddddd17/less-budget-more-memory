# Use-site rematerialization: prototype for pytorch/pytorch#197838

`use_site_remat.py` is a self-contained version of the pass evaluated in the paper. It makes the same edits as
`experiments/liveness/remat_at_use.py`, which produced the paper's numbers; a test checks that they match.

**What it fixes.** Under `activation_memory_budget`, AOTAutograd recomputes each value at most once in the backward,
at its first use, and holds it until its last use. At small budgets many recomputed values are held across the
backward peak, so the peak rises as the budget falls. The pass runs after partitioning. When a value is live at the
peak and used again after it, the pass gives those later uses their own recomputation just before they need it. The
saved set is never changed.

**Usage (prototype hook, Inductor):**

```python
import torch, use_site_remat
torch._functorch.config.activation_memory_budget = 0.05
with use_site_remat.enabled("pointwise"):        # or "all": may also duplicate matmuls
    torch.compile(model)(x).backward()           # the first (compiling) call must be inside the context
print(use_site_remat.LAST_STATS[-1])             # static peak before/after, edits accepted, random-op counts
```

For a custom `aot_autograd` backend, pass `partition_fn=use_site_remat.make_partition_fn("pointwise")`.

| File | What it is |
|---|---|
| `use_site_remat.py` | the pass and the hooks |
| `test_use_site_remat.py` | CPU tests (`pytest prototype/test_use_site_remat.py`, ~2 min). They check that the saved set is unchanged, gradients are identical, the static peak never rises, random ops are never duplicated (dropout model), the edits are deterministic, and the result matches the paper implementation. |
| `demo_issue_197838.py` | the issue's model on CUDA: stock vs both modes, peak, step time and gradient error vs eager |

**Limits.** The pass's peak model is a static walk of the backward graph. It is exact when the graph runs as emitted,
but only approximate under Inductor, which fuses ops and reuses buffers. Other limits:
- tested only with torch 2.14, fp32, small models and one consumer GPU;
- no distributed, mutation-heavy or FSDP graphs;
- the search re-walks the whole graph after every accepted edit (quadratic in graph size): 1-48 s per graph for the
  paper's models, and up to +56% total compile time under Inductor;
- attention is never duplicated, even in "all" mode (SDPA ops carry the `nondeterministic_seeded` tag).

This is research code, not production code.

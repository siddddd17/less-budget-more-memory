"""Probe (CPU, 4-layer HF BERT, Inductor): does the backward depend on the global RNG state?
Advancing the generator between forward and backward must not change gradients if the backward
replays the forward's randomness. Used for the 3.7% vs 1.5e-7 numbers in Section 6 of the paper.
Run from experiments/liveness/: python rng_redraw_test.py"""
import sys; sys.argv=['x']
import repro_rng_bert_cpu as RB, torch
import torch._functorch.config as fc
for b in (1.0, 0.05):
    torch._dynamo.reset()
    w = RB.build(4); ids = torch.randint(0, 1000, (2, 256), generator=torch.Generator().manual_seed(3))
    cm = torch.compile(w, backend="inductor", dynamic=False)
    fc.activation_memory_budget = b
    torch.manual_seed(1234); cm(ids).backward()
    gs = []
    for burn in (0, 1):
        w.zero_grad(set_to_none=True)
        torch.manual_seed(1234); L = cm(ids)
        if burn: torch.rand(1000)          # advance the global generator between forward and backward
        L.backward()
        gs.append([p.grad.detach().clone() for p in w.parameters() if p.grad is not None])
    fc.activation_memory_budget = 1.0
    d = sum(float((x - y).pow(2).sum()) for x, y in zip(*gs)) ** 0.5 / sum(float(x.pow(2).sum()) for x in gs[0]) ** 0.5
    print(f"budget {b}: grad change when RNG advanced between fwd and bwd = {d:.2e}  bw_rng={RB.NAMES.get('bw')}")

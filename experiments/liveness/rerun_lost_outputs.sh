#!/usr/bin/env bash
# rerun_lost_outputs.sh -- regenerate, as raw files, the results that previously existed only as transcribed terminal
# output (results/LOGGED_OUTPUT.md), plus CPU controls and the environment record. Run from the repository root on the
# GPU machine, inside tmux:   bash experiments/liveness/rerun_lost_outputs.sh
# Total time on a GTX 1650: roughly 1.5-2 hours (GPU ~1.1-1.4 h, CPU ~30 min). Each step writes its own log, so a failed step can be rerun alone.
set -u
export PYTHONPATH="$PWD" TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
L=experiments/liveness; R=$L/results
FAILED=()
run() { local name="$1"; shift; echo "=== $name: $*"; "$@" 2>&1 | tee "$R/${name}_log.txt"; local rc=${PIPESTATUS[0]}
        echo "=== $name exit $rc"; [ "$rc" -eq 0 ] || FAILED+=("$name"); }

# environment record
{ date -Is; python -V; pip freeze | grep -i -E "^(torch|transformers|triton|numpy|matplotlib|sympy)"; \
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'device',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"; \
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null; uname -a; } > "$R/environment.txt" 2>&1

# GPU: Experiment A (aot_eager oracle, 14 Llama plans + BERT), Experiment C part 1 (Inductor oracle), Llama vs eager
run expa_full     python $L/liveness_oracle_check.py --suite full --out $R/expa_full.json
run expc_oracle_full python $L/experiment_c_inductor_runtime.py --part oracle --suite full --out $R/expc_oracle_full.json
run check_grads   python $L/check_grads.py

# CPU: RNG side finding (Section 6) and the fix-disabled control (Section 5.3)
run repro_rng_budget_inductor_cpu python $L/repro_rng_budget.py --device cpu --backend inductor --budgets 1.0 0.5 0.1 0.0
run repro_rng_budget_aot_eager_cpu python $L/repro_rng_budget.py --device cpu --backend aot_eager --budgets 1.0 0.9 0.5 0.1 0.0
run rng_redraw_test_cpu bash -c "cd $L && python rng_redraw_test.py"
run repro_rng_bert_cpu_none python $L/repro_rng_bert_cpu.py --mode none
run repro_rng_bert_cpu_v2   python $L/repro_rng_bert_cpu.py --mode v2
run check_remat_bert_nofix_cpu env CHECK_NO_FIX=1 CHECK_DEVICE=cpu CHECK_SCALE=1 CHECK_BUDGETS=0.05,0.10,0.15 python $L/check_remat_bert.py
run rng_fix_v1_v2_equiv python $L/check_rng_fix_v1_v2_equiv.py --scale 8 --device cpu
if [ ${#FAILED[@]} -eq 0 ]; then echo "ALL DONE: every step succeeded"; else echo "DONE WITH FAILURES: ${FAILED[*]}"; fi

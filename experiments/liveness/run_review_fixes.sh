#!/usr/bin/env bash
# run_review_fixes.sh -- GPU runs requested by the internal review (3 Oct 2026). Run AFTER the check_grads loop finishes.
#   1 BERT Inductor rows with the fixed remat pass (random ops never duplicated)       (~60-75 min)
#   2 BERT gradient checks with the fixed pass (dropout on, fix emulated)                (~25-35 min)
#   3 dense stock grid, Llama + BERT, Inductor                                           (~60-75 min)
# From the ackaudit root, inside tmux, with the CUDA venv active:  bash experiments/liveness/run_review_fixes.sh
set -u
export PYTHONPATH="$PWD" TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}" TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
L=experiments/liveness; R=$L/results; FAILED=()
run() { local name="$1"; shift; echo "=== $(date +%H:%M) $name"; "$@" 2>&1 | grep -v -i "warn" | tee "$R/${name}_log.txt"; local rc=${PIPESTATUS[0]}
        echo "=== $(date +%H:%M) $name exit $rc" | tee -a "$R/${name}_log.txt"; [ "$rc" -eq 0 ] || FAILED+=("$name"); }
run gpu_eval_remat_bert_fixed_v3 python $L/gpu_eval_remat.py --suite bert --emulate-rng-fix --backends inductor --out $R/gpu_eval_remat_bert_fixed_v3.json
run check_remat_bert_v3          python $L/check_remat_bert.py
run check_remat_bert_b2_v3       env CHECK_BUDGETS=0.10,0.15 python $L/check_remat_bert.py
run stock_grid                   python $L/stock_grid.py --out $R/stock_grid.json
grep -h "VERDICT\|random_ops\|lowest stock" $R/check_remat_bert_v3_log.txt $R/check_remat_bert_b2_v3_log.txt $R/stock_grid_log.txt
if [ ${#FAILED[@]} -eq 0 ]; then echo "ALL DONE"; else echo "DONE WITH FAILURES: ${FAILED[*]}"; fi

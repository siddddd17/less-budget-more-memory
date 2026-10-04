#!/usr/bin/env bash
# run_mlsys_gpu.sh -- the two GPU experiments added after the internal review:
#   1 seqlen : sequence-length / MLP-width sweep of the stock curve under Inductor        (~2 h)
#   2 edits  : remat pass, predicted vs measured peak for prefixes of its edits (Inductor)  (~1 h)
# Run from the repo root, inside tmux:   bash experiments/liveness/run_mlsys_gpu.sh
set -u
export PYTHONPATH="$PWD" TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
D=experiments/liveness; R=$D/results; mkdir -p "$R"
FAILED=()
run() { local name="$1"; shift; echo "=== $(date +%H:%M) $name"; "$@" 2>&1 | tee "$R/${name}_log.txt"; local rc=${PIPESTATUS[0]}
        echo "=== $(date +%H:%M) $name exit $rc" | tee -a "$R/${name}_log.txt"; [ "$rc" -eq 0 ] || FAILED+=("$name"); }
{ date -Is; python -V; python -c "import torch,triton,transformers;print('torch',torch.__version__,'triton',triton.__version__,'transformers',transformers.__version__,'cuda',torch.version.cuda,torch.cuda.get_device_name(0))"; } > "$R/environment_mlsys.txt" 2>&1

run seqlen_sweep_gpu        python $D/seqlen_sweep.py --device cuda --out $R/seqlen_sweep_gpu.json
run remat_edit_validation   python $D/remat_edit_validation.py --out $R/remat_edit_validation.json
grep -h "^== " $R/remat_edit_validation_log.txt
if [ ${#FAILED[@]} -eq 0 ]; then echo "ALL DONE"; else echo "DONE WITH FAILURES: ${FAILED[*]}"; fi

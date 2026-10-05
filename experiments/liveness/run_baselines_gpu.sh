#!/usr/bin/env bash
# run_baselines_gpu.sh -- practitioner activation-checkpointing baselines (per-layer AC, every-2nd-layer AC, torchtitan
# op-SAC, save-all-matmuls SAC; eager and compiled) next to the paper's reference points.  ~1.5-2 h on the GTX 1650.
# Run from the repo root, inside tmux:   bash experiments/liveness/run_baselines_gpu.sh
set -u
export PYTHONPATH="$PWD" TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
D=experiments/liveness; R=$D/results; mkdir -p "$R"
{ date -Is; python -V; python -c "import torch,triton,transformers;print('torch',torch.__version__,'triton',triton.__version__,'transformers',transformers.__version__,'cuda',torch.version.cuda,torch.cuda.get_device_name(0))"; } > "$R/environment_baselines.txt" 2>&1
echo "=== $(date +%H:%M) ac_baselines"
python $D/ac_baselines.py --out $R/ac_baselines.json 2>&1 | tee $R/ac_baselines_log.txt
rc=${PIPESTATUS[0]}; echo "=== $(date +%H:%M) ac_baselines exit $rc" | tee -a $R/ac_baselines_log.txt

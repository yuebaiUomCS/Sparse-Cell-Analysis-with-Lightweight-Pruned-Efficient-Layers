#!/usr/bin/env bash
set -euo pipefail

# Run only Stage-B round1 (train Bayes head, NO pruning)
# Usage:
#   ./stageb_round1.sh            # use defaults from script
#   START_CKPT=/path/to/A_best ./stageb_round1.sh
#   CV_SPLITS_B=1 EPOCHS=50 ./stageb_round1.sh

# --- Defaults (kept in sync with stageb.sh) ---
ENV_NAME="scbert"
PY=python
NPROC=1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}

DATA="/home/linux/scBERT/data/STARmap_merged_fixed.h5ad"
OUTROOT="final_ab_run"
OUT_B="${OUTROOT}/stageB"
mkdir -p "${OUT_B}"

# model/data params
GENE_NUM=166
BIN_NUM=5
USE_POSEMB=1

# Stage-A best (fallback start ckpt) - override by setting START_CKPT env var
A_BEST="/home/linux/scBERT/final_ab_run/stageAstageA_A_det_do0.5_l11e-41_best.pth"
START_CKPT=${START_CKPT:-${A_BEST}}

# Training hyperparams (override via env)
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-4}
GRAD_ACC=${GRAD_ACC:-60}
LR=${LR:-1.5e-4}
CV_SPLITS_B=${CV_SPLITS_B:-1}

# Bayes settings
USE_BAYES=${USE_BAYES:-1}
KL_WEIGHT=${KL_WEIGHT:-1e-6}
KL_STEPS=${KL_STEPS:-10000}
MC_SAMPLES=${MC_SAMPLES:-1}
MC_SAMPLES_EVAL=${MC_SAMPLES_EVAL:-8}
PRE_WARM=${PRE_WARM:-6}

# Regularization / head lr
B_DROPOUT=${B_DROPOUT:-0.5}
B_L1=${B_L1:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
HEAD_LR_MULT=${HEAD_LR_MULT:-2.0}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-15}

# Positional embedding flag
POSEMB_FLAG=()
if [[ "${USE_POSEMB}" == "1" ]]; then
  POSEMB_FLAG+=(--pos_embed)
fi

# Launcher
if command -v torchrun >/dev/null 2>&1; then
  if [[ "${NPROC}" -gt 1 ]]; then
    LAUNCH_CMD=(torchrun --nproc_per_node="${NPROC}")
  else
    LAUNCH_CMD=("${PY}")
  fi
else
  LAUNCH_CMD=("${PY}" -m torch.distributed.launch --nproc_per_node="${NPROC}")
fi

if [[ ! -f "${START_CKPT}" ]]; then
  echo "[ERR] START_CKPT not found: ${START_CKPT}"
  echo "请设置 START_CKPT=/path/to/A_best 或先运行 Stage-A。"
  exit 1
fi

B_TAG_R1="B_round1_bayes_do${B_DROPOUT}_l1${B_L1}_kw${KL_WEIGHT}_ks${KL_STEPS}_mc${MC_SAMPLES}_pw${PRE_WARM}"
B_MODEL_NAME_R1="stageB_${B_TAG_R1}"
B_LOG_R1="${OUT_B}/${B_MODEL_NAME_R1}.log"

# Save config JSON
"${PY}" - <<PY "${OUT_B}/stageB_round1_config.json" stage="B_round1" data_path="${DATA}" init_ckpt="${START_CKPT}" gene_num=${GENE_NUM} bin_num=${BIN_NUM} pos_embed=${USE_POSEMB} dropout=${B_DROPOUT} l1_lambda=${B_L1} epochs=${EPOCHS} batch_size=${BATCH} grad_acc=${GRAD_ACC} learning_rate=${LR} cv_splits=${CV_SPLITS_B} use_bayes=true kl_weight=${KL_WEIGHT} kl_anneal_steps=${KL_STEPS} bayes_mc_samples_train=${MC_SAMPLES} bayes_mc_samples_eval=${MC_SAMPLES_EVAL} pre_warm_epochs=${PRE_WARM} weight_decay=${WEIGHT_DECAY} head_lr_mult=${HEAD_LR_MULT} patience=${EARLY_STOP_PATIENCE}
import json,sys
out=sys.argv[1]
pairs=sys.argv[2:]
cfg={}
for p in pairs:
    if '=' not in p: continue
    k,v=p.split('=',1)
    try:
        if v.lower() in ('true','false'):
            vv=(v.lower()=='true')
        else:
            vv=eval(v,{},{})
    except Exception:
        vv=v
    cfg[k]=vv
with open(out,'w') as f:
    json.dump(cfg,f,indent=2,ensure_ascii=False)
print(f"[OK] wrote {out}")
PY

# Build args and run
B_ARGS_R1=(
  --data_path "${DATA}"
  --model_path "${START_CKPT}"
  --ckpt_dir "${OUT_B}"
  --model_name "${B_MODEL_NAME_R1}"
  --batch_size "${BATCH}"
  --epoch "${EPOCHS}"
  --gene_num "${GENE_NUM}"
  --bin_num "${BIN_NUM}"
  --learning_rate "${LR}"
  --grad_acc "${GRAD_ACC}"
  --cv_splits "${CV_SPLITS_B}"
  --dropout "${B_DROPOUT}"
  --l1_lambda "${B_L1}"
  --weight_decay "${WEIGHT_DECAY}"
  --pre_warm_epochs "${PRE_WARM}"
  --head_lr_mult "${HEAD_LR_MULT}"
  --patience "${EARLY_STOP_PATIENCE}"
  "${POSEMB_FLAG[@]}"
)
if [[ "${USE_BAYES}" == "1" ]]; then
  B_ARGS_R1+=(--bayes_head --kl_weight "${KL_WEIGHT}" --kl_anneal_steps "${KL_STEPS}" --bayes_mc_samples "${MC_SAMPLES_EVAL}")
fi

echo ">>> Running Stage-B round1 -> ${B_MODEL_NAME_R1}"
"${LAUNCH_CMD[@]}" finetune_sparse_stageB.py "${B_ARGS_R1[@]}" 2>&1 | tee -- "${B_LOG_R1}"

echo "\n[DONE] round1 finished. 若已生成 best ckpt，请将其路径作为 START_CKPT 传给 round2。示例："
echo "  START_CKPT=/path/to/round1_best.pth ./stageb_round2.sh"

exit 0

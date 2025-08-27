#!/usr/bin/env bash
set -euo pipefail

# Run only Stage-B round2 (secondary pruning + fine-tune)
# Usage:
#   START_CKPT=/path/to/round1_best.pth ./stageb_round2.sh
#   PRUNE_SECONDARY_TARGET=0.3 ./stageb_round2.sh

# --- Defaults (kept in sync with stageb.sh) ---
ENV_NAME="scbert"
PY=python
NPROC=1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}

DATA="/home/linux/scBERT/data/STARmap_merged_fixed.h5ad"
OUTROOT="final_ab_run"
OUT_B="${OUTROOT}/stageB"
mkdir -p "${OUT_B}"

# START_CKPT must be provided (round1_best or A_BEST fallback)
A_BEST="/home/linux/scBERT/final_ab_run/stageAstageA_A_det_do0.5_l11e-41_best.pth"
START_CKPT=/home/linux/scBERT/final_ab_run/stageBstageB_B_round1_bayes_do0.5_l10_kw1e-6_ks10000_mc1_pw61_best.pth
if [[ -z "${START_CKPT}" ]]; then
  # Try to auto-find the most recent *_best.pth produced by round1 in the stageB output dir
  BEST_CAND=$(ls -1t "${OUT_B}"/*_best.pth 2>/dev/null | head -n1 || true)
  if [[ -n "${BEST_CAND}" && -f "${BEST_CAND}" ]]; then
    echo "[INFO] No START_CKPT provided. Auto-using latest best checkpoint: ${BEST_CAND}"
    START_CKPT="${BEST_CAND}"
  else
    echo "[ERR] Please provide START_CKPT=/path/to/round1_best.pth"
    echo "Example: START_CKPT=${A_BEST} ./stageb_round2.sh"
    exit 1
  fi
fi

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

# Pruning params
PRUNE_SECONDARY_TARGET=${PRUNE_SECONDARY_TARGET:-0.25}
PRUNE_SECONDARY_STEP=${PRUNE_SECONDARY_STEP:-0.05}
PRUNE_SECONDARY_FT_EPOCHS=${PRUNE_SECONDARY_FT_EPOCHS:-4}
PRUNE_SECONDARY_F1_TOL=${PRUNE_SECONDARY_F1_TOL:-0.005}
PRUNE_SECONDARY_LAYERS=${PRUNE_SECONDARY_LAYERS:-"fc1"}
PRUNE_DIM=${PRUNE_DIM:-0}

# Positional embedding flag
USE_POSEMB=${USE_POSEMB:-1}
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
  exit 1
fi

B_TAG_R2="B_round2_prune_do${B_DROPOUT}_l1${B_L1}_t${PRUNE_SECONDARY_TARGET}_s${PRUNE_SECONDARY_STEP}_ft${PRUNE_SECONDARY_FT_EPOCHS}_d${PRUNE_DIM}_L$(echo ${PRUNE_SECONDARY_LAYERS} | tr ',' '-')_kw${KL_WEIGHT}_ks${KL_STEPS}_mc${MC_SAMPLES}_pw${PRE_WARM}"
B_MODEL_NAME_R2="stageB_${B_TAG_R2}"
B_LOG_R2="${OUT_B}/${B_MODEL_NAME_R2}.log"

# Save config JSON
"${PY}" - <<PY "${OUT_B}/stageB_round2_config.json" stage="B_round2" data_path="${DATA}" init_ckpt="${START_CKPT}" gene_num=166 bin_num=5 pos_embed=${USE_POSEMB} dropout=${B_DROPOUT} l1_lambda=${B_L1} epochs=${EPOCHS} batch_size=${BATCH} grad_acc=${GRAD_ACC} learning_rate=${LR} cv_splits=${CV_SPLITS_B} use_bayes=true kl_weight=${KL_WEIGHT} kl_anneal_steps=${KL_STEPS} bayes_mc_samples_train=${MC_SAMPLES} bayes_mc_samples_eval=${MC_SAMPLES_EVAL} pre_warm_epochs=${PRE_WARM} weight_decay=${WEIGHT_DECAY} head_lr_mult=${HEAD_LR_MULT} patience=${EARLY_STOP_PATIENCE} prune_target=${PRUNE_SECONDARY_TARGET} prune_step=${PRUNE_SECONDARY_STEP} prune_ft_epochs=${PRUNE_SECONDARY_FT_EPOCHS} prune_f1_tol=${PRUNE_SECONDARY_F1_TOL} prune_dim=${PRUNE_DIM} prune_layers="${PRUNE_SECONDARY_LAYERS}"
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
B_ARGS_R2=(
  --data_path "${DATA}"
  --model_path "${START_CKPT}"
  --ckpt_dir "${OUT_B}"
  --model_name "${B_MODEL_NAME_R2}"
  --batch_size "${BATCH}"
  --epoch "${EPOCHS}"
  --gene_num "166"
  --bin_num "5"
  --learning_rate "${LR}"
  --grad_acc "${GRAD_ACC}"
  --cv_splits "${CV_SPLITS_B}"
  --dropout "${B_DROPOUT}"
  --l1_lambda "${B_L1}"
  --weight_decay "${WEIGHT_DECAY}"
  --prune_target "${PRUNE_SECONDARY_TARGET}"
  --prune_step "${PRUNE_SECONDARY_STEP}"
  --prune_ft_epochs "${PRUNE_SECONDARY_FT_EPOCHS}"
  --prune_f1_tol "${PRUNE_SECONDARY_F1_TOL}"
  --prune_dim "${PRUNE_DIM}"
  --prune_layers "${PRUNE_SECONDARY_LAYERS}"
  --pre_warm_epochs "${PRE_WARM}"
  --head_lr_mult "${HEAD_LR_MULT}"
  --patience "${EARLY_STOP_PATIENCE}"
  "${POSEMB_FLAG[@]}"
)
if [[ "${USE_BAYES}" == "1" ]]; then
  B_ARGS_R2+=(--bayes_head --kl_weight "${KL_WEIGHT}" --kl_anneal_steps "${KL_STEPS}" --bayes_mc_samples "${MC_SAMPLES_EVAL}")
fi

echo ">>> Running Stage-B round2 (prune) -> ${B_MODEL_NAME_R2}"
"${LAUNCH_CMD[@]}" finetune_sparse_stageB.py "${B_ARGS_R2[@]}" 2>&1 | tee -- "${B_LOG_R2}"

exit 0

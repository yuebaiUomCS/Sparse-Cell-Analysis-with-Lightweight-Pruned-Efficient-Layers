#!/usr/bin/env bash
set -euo pipefail

############################################
# 环境与通用设置（按需修改）
############################################
ENV_NAME="scbert"
PY=python
NPROC=1                           # 单卡=1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}

DATA="/home/linux/scBERT/data/STARmap_merged_fixed.h5ad"
PRETRAIN_CKPT="panglao_pretrain.pth"       # 预训练骨干
OUTROOT="final_ab_run"                      # 整次实验输出根目录
OUT_A="${OUTROOT}/stageA"
OUT_B="${OUTROOT}/stageB"
mkdir -p "${OUT_A}" "${OUT_B}"

GENE_NUM=166
BIN_NUM=5
USE_POSEMB=1

# ====== Stage-A 最佳组合（你给的最佳行）======
A_DROPOUT=0.5
A_L1=1e-4

# ====== Stage-B 建议（Bayes + 更弱 L1）======
B_DROPOUT=0.5
B_L1=1e-6

# 训练 epoch / batch 等
EPOCHS=100
BATCH=4
GRAD_ACC=60
LR=1.5e-4
CV_SPLITS_A=1
CV_SPLITS_B=1

# Bayes & 剪枝（阶段B用）
USE_BAYES=1
KL_WEIGHT=1e-6
KL_STEPS=10000
# 训练期轻量 mc=1；评估用更高采样（仅写入配置）
MC_SAMPLES=1
MC_SAMPLES_EVAL=8
PRE_WARM=6

# Head / 正则化
# 建议：关闭额外 L1，使用轻微 weight decay；头部 lr 倍数可传入（若 finetune 支持）
B_L1=0
WEIGHT_DECAY=1e-4
HEAD_LR_MULT=2.0

# Optional: 强制指定用于 round2 的起始 ckpt（覆盖自动选择）。设置为完整路径以启用，否则留空。
# 示例（请按需修改）：
# HARDCODE_ROUND2_CKPT="/home/linux/scBERT/final_ab_run/stageBstageB_B_round1_bayes_do0.5_l10_t0.0_s0.05_ft4_d0_Lfc1_kw1e-6_ks10000_mc1_pw61_best.pth"
# HARDCODE_ROUND2_CKPT=""
# Hardcoded round2 start checkpoint (forced override) — set to provided path
HARDCODE_ROUND2_CKPT="/home/linux/scBERT/final_ab_run/stageBstageB_B_bayes_do0.5_l11e-6_t0.4_s0.1_ft2_d0_Lfc1-fc2_kw5e-6_ks10000_mc8_pw21_best.pth"

# 如果想直接跳过 round1（只跑 round2），将此变量设为 1。脚本会尝试使用 HARDCODE_ROUND2_CKPT（若存在）作起始 ckpt，
# 否则会在进入 round2 的时候回退到 A_BEST 或尝试找到最新的 round1 ckpt。
SKIP_ROUND1=${SKIP_ROUND1:-0}

# Early stopping patience (epochs without val improvement before stop)
EARLY_STOP_PATIENCE=15

# Skip round1 and directly run round2 (use HARDCODE_ROUND2_CKPT or A_BEST as start)
# Set to 1 to bypass round1 training entirely and jump straight to round2 pruning.
# SKIP_ROUND1=1

# 剪枝（默认第一轮不剪，第二轮可选渐进剪）
# 第一轮（round1）默认不剪，用于稳定 Bayes 头
PRUNE_TARGET=0.0
PRUNE_STEP=0.05
PRUNE_FT_EPOCHS=4
PRUNE_F1_TOL=0.005
PRUNE_DIM=0
PRUNE_LAYERS="fc1"

# 二轮剪枝开关（若为1，则在 round1 的 best ckpt 基础上做渐进剪枝）
RUN_PRUNE_SECONDARY=1
PRUNE_SECONDARY_TARGET=0.25
PRUNE_SECONDARY_STEP=0.05
PRUNE_SECONDARY_FT_EPOCHS=4
PRUNE_SECONDARY_F1_TOL=0.005
PRUNE_SECONDARY_LAYERS="fc1"

# 如果不想分两轮（round1 + round2），而是一次性在同一训练中完成 Bayes 头训练与渐进剪枝（合并模式），可开启此标志。
# 当 MERGE_ROUNDS=1 时，脚本会跳过 round1 并直接运行一个带有二轮剪枝目标的 Stage-B 训练（init_ckpt = HARDCODE_ROUND2_CKPT 或 A_BEST）。
MERGE_ROUNDS=${MERGE_ROUNDS:-0}

############################################
# Conda & 启动器
############################################
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERR] conda 未找到，请先 'source ~/anaconda3/etc/profile.d/conda.sh'"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "[INFO] Activated conda env: ${ENV_NAME}"
[[ -n "${CUDA_VISIBLE_DEVICES}" ]] && echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export LOCAL_RANK=0

if command -v torchrun >/dev/null 2>&1; then
  if [[ "${NPROC}" -gt 1 ]]; then
    LAUNCH_CMD=(torchrun --nproc_per_node="${NPROC}")
    echo "[INFO] Using torchrun (nproc_per_node=${NPROC})"
  else
    LAUNCH_CMD=("${PY}")
    echo "[INFO] Using plain python (single-process)"
  fi
else
  echo "[INFO] torchrun not found, fallback to 'python -m torch.distributed.launch'"
  LAUNCH_CMD=("${PY}" -m torch.distributed.launch --nproc_per_node="${NPROC}")
fi

POSEMB_FLAG=()
if [[ "${USE_POSEMB}" == "1" ]]; then
  POSEMB_FLAG+=(--pos_embed)
fi

############################################
# 工具：保存配置为 JSON（不丢参数）
############################################
save_json () {
  # $1: 输出 JSON 路径；其余参数格式 k=v
  local out_json="$1"; shift
  "${PY}" - <<'PY' "$out_json" "$@"
import json, sys
out = sys.argv[1]
pairs = sys.argv[2:]
cfg = {}
for p in pairs:
    if "=" not in p: 
        continue
    k,v = p.split("=",1)
    # 尝试把数值/布尔转成真类型；字符串保持原样
    try:
        if v.lower() in ("true","false"):
            vv = (v.lower()=="true")
        else:
            vv = eval(v, {}, {})
    except Exception:
        vv = v
    cfg[k]=vv
with open(out, "w") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print(f"[OK] wrote {out}")
PY
}

############################################
# 小工具：合并CSV（首文件保留表头）
############################################
merge_csv() {
  # $1: 搜索模式；$2: 输出文件
  local pattern="$1"
  local out="$2"
  local first=1
  : > "${out}"
  for f in ${pattern}; do
    [[ -s "$f" ]] || continue
    if [[ $first -eq 1 ]]; then
      cat "$f" >> "${out}"
      first=0
    else
      tail -n +2 "$f" >> "${out}"
    fi
  done
  echo "[OK] merged -> ${out}"
}

############################################
# 阶段A：确定性头（不Bayes、不剪枝） -- SKIPPED
############################################
A_TAG="A_det_do${A_DROPOUT}_l1${A_L1}"
A_MODEL_NAME="stageA_${A_TAG}"
A_LOG="${OUT_A}/${A_MODEL_NAME}.log"

echo ">>> [A] Stage-A skipped: using existing checkpoint for Stage-B."

# 保留 A 相关变量供后续汇总/日志使用，但不生成配置或运行训练（避免覆盖已有结果）
A_ARGS=()
if [[ "${LAUNCH_CMD[*]}" == *"torch.distributed.launch"* ]]; then
  A_ARGS=(--local_rank 0 "${A_ARGS[@]}")
fi

# Skip Stage-A training: use an existing A best checkpoint instead of running training here
echo ">>> [A] Skipping Stage-A training; using existing checkpoint for Stage-B and pretrain: /home/linux/scBERT/final_ab_run/stageAstageA_A_det_do0.5_l11e-41_best.pth"
# "${LAUNCH_CMD[@]}" finetune_sparse_stageB.py "${A_ARGS[@]}" 2>&1 | tee -- "${A_LOG}"

# 阶段A best ckpt（强制使用指定路径，跳过自动查找）
A_BEST="/home/linux/scBERT/final_ab_run/stageAstageA_A_det_do0.5_l11e-41_best.pth"
if [[ ! -f "${A_BEST}" ]]; then
  echo "[WARN] 强制指定的 A_BEST 未找到：${A_BEST}。请确认路径或将 PRETRAIN_CKPT 用作后备。"
fi
echo "[INFO] 使用 Stage-A 初始化权重：${A_BEST}"

############################################
# 阶段B：Bayes + 渐进式结构化剪枝（两轮：先不剪稳定 Bayes；可选二轮渐进剪枝）
############################################
B_TAG_R1="B_round1_bayes_do${B_DROPOUT}_l1${B_L1}_t${PRUNE_TARGET}_s${PRUNE_STEP}_ft${PRUNE_FT_EPOCHS}_d${PRUNE_DIM}_L$(echo ${PRUNE_LAYERS} | tr ',' '-')_kw${KL_WEIGHT}_ks${KL_STEPS}_mc${MC_SAMPLES}_pw${PRE_WARM}"
B_MODEL_NAME_R1="stageB_${B_TAG_R1}"
B_LOG_R1="${OUT_B}/${B_MODEL_NAME_R1}.log"

echo ">>> [B][round1] Train Bayes head (NO pruning): ${B_MODEL_NAME_R1}"

# 保存 round1 配置
save_json "${OUT_B}/stageB_round1_config.json" \
  stage="B_round1" data_path="${DATA}" init_ckpt="${A_BEST}" \
  gene_num=${GENE_NUM} bin_num=${BIN_NUM} pos_embed=${USE_POSEMB} \
  dropout=${B_DROPOUT} l1_lambda=${B_L1} \
  epochs=${EPOCHS} batch_size=${BATCH} grad_acc=${GRAD_ACC} learning_rate=${LR} \
  cv_splits=${CV_SPLITS_B} use_bayes=true \
  kl_weight=${KL_WEIGHT} kl_anneal_steps=${KL_STEPS} bayes_mc_samples_train=${MC_SAMPLES} bayes_mc_samples_eval=${MC_SAMPLES_EVAL} pre_warm_epochs=${PRE_WARM} \
  weight_decay=${WEIGHT_DECAY} head_lr_mult=${HEAD_LR_MULT} patience=${EARLY_STOP_PATIENCE} \
  prune_target=${PRUNE_TARGET} prune_step=${PRUNE_STEP} prune_ft_epochs=${PRUNE_FT_EPOCHS} \
  prune_f1_tol=${PRUNE_F1_TOL} prune_dim=${PRUNE_DIM} prune_layers="${PRUNE_LAYERS}"

B_ARGS_R1=(
  --data_path "${DATA}"
  --model_path "${A_BEST}"
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
  --prune_target "${PRUNE_TARGET}"
  --prune_step "${PRUNE_STEP}"
  --prune_ft_epochs "${PRUNE_FT_EPOCHS}"
  --prune_f1_tol "${PRUNE_F1_TOL}"
  --prune_dim "${PRUNE_DIM}"
  --prune_layers "${PRUNE_LAYERS}"
  --pre_warm_epochs "${PRE_WARM}"
  --head_lr_mult "${HEAD_LR_MULT}"
  --patience "${EARLY_STOP_PATIENCE}"
  "${POSEMB_FLAG[@]}"
)
if [[ "${USE_BAYES}" == "1" ]]; then
  B_ARGS_R1+=(--bayes_head --kl_weight "${KL_WEIGHT}" --kl_anneal_steps "${KL_STEPS}" --bayes_mc_samples "${MC_SAMPLES_EVAL}")
fi
if [[ "${LAUNCH_CMD[*]}" == *"torch.distributed.launch"* ]]; then
  B_ARGS_R1=(--local_rank 0 "${B_ARGS_R1[@]}")
fi

# Run or skip round1 depending on SKIP_ROUND1. If skipping, prefer HARDCODE_ROUND2_CKPT as the starting ckpt for round2.
if [[ "${SKIP_ROUND1}" == "1" ]]; then
  echo ">>> [B] SKIP_ROUND1=1 -> skipping Round1 training and proceeding to Round2."
  if [[ -n "${HARDCODE_ROUND2_CKPT:-}" && -f "${HARDCODE_ROUND2_CKPT}" ]]; then
    B_BEST_R1="${HARDCODE_ROUND2_CKPT}"
    echo "[INFO] Using HARDCODE_ROUND2_CKPT as round1 best: ${B_BEST_R1}"
  else
    B_BEST_R1=""
    echo "[INFO] No HARDCODE_ROUND2_CKPT provided/found. Round2 will attempt fallback to latest round1 ckpt or A_BEST."
  fi
else
  "${LAUNCH_CMD[@]}" finetune_sparse_stageB.py "${B_ARGS_R1[@]}" 2>&1 | tee -- "${B_LOG_R1}"
  # 找到 round1 的 best ckpt（约定名：<model_name>_best.pth）
  B_BEST_R1="${OUT_B}/${B_MODEL_NAME_R1}_best.pth"
  if [[ ! -f "${B_BEST_R1}" ]]; then
    echo "[WARN] Round1 best checkpoint 未找到：${B_BEST_R1}。后续若开启二轮剪枝将尝试使用最后保存的 ckpt。"
  fi
fi

# 若启用二轮剪枝，则以 round1 的 best ckpt 为初始化，进行渐进剪枝
B_MODEL_NAME="${B_MODEL_NAME_R1}"
B_LOG="${B_LOG_R1}"
if [[ "${RUN_PRUNE_SECONDARY}" == "1" ]]; then
  echo ">>> [B][round2] Secondary pruning run enabled -> target ${PRUNE_SECONDARY_TARGET}"
  # 确保使用 round1 的 best ckpt（若不存在则使用 round1 最后保存模型）
  # 默认以 round1 best 作为起始 ckpt，除非启用了 HARDCODE_ROUND2_CKPT
  START_CKPT="${B_BEST_R1}"
  if [[ -n "${HARDCODE_ROUND2_CKPT:-}" ]]; then
    if [[ -f "${HARDCODE_ROUND2_CKPT}" ]]; then
      echo "[INFO] HARDCODE_ROUND2_CKPT is set -> forcing START_CKPT to ${HARDCODE_ROUND2_CKPT}"
      START_CKPT="${HARDCODE_ROUND2_CKPT}"
    else
      echo "[WARN] HARDCODE_ROUND2_CKPT set but file not found: ${HARDCODE_ROUND2_CKPT}. Falling back to normal logic."
    fi
  fi
  if [[ ! -f "${START_CKPT}" ]]; then
    # try fallback to pattern match for any ckpt with model_name prefix
    START_CKPT="$(ls -1t ${OUT_B}/${B_MODEL_NAME_R1}*.pth 2>/dev/null | head -n1 || true)"
    if [[ -z "${START_CKPT}" ]]; then
      echo "[WARN] 未找到 round1 ckpt，尝试回退到 Stage-A 指定的 A_BEST：${A_BEST}"
      if [[ -f "${A_BEST}" ]]; then
        START_CKPT="${A_BEST}"
        echo "[INFO] 使用回退 ckpt -> ${START_CKPT}"
      else
        echo "[WARN] 回退的 A_BEST 也不存在：${A_BEST}。将跳过 round2。"
      fi
    else
      echo "[INFO] Using latest round1 checkpoint: ${START_CKPT}"
    fi
  fi
  if [[ -n "${START_CKPT}" && -f "${START_CKPT}" ]]; then
    B_TAG_R2="B_round2_prune_do${B_DROPOUT}_l1${B_L1}_t${PRUNE_SECONDARY_TARGET}_s${PRUNE_SECONDARY_STEP}_ft${PRUNE_SECONDARY_FT_EPOCHS}_d${PRUNE_DIM}_L$(echo ${PRUNE_SECONDARY_LAYERS} | tr ',' '-')_kw${KL_WEIGHT}_ks${KL_STEPS}_mc${MC_SAMPLES}_pw${PRE_WARM}"
    B_MODEL_NAME_R2="stageB_${B_TAG_R2}"
    B_LOG_R2="${OUT_B}/${B_MODEL_NAME_R2}.log"

    save_json "${OUT_B}/stageB_round2_config.json" \
      stage="B_round2" data_path="${DATA}" init_ckpt="${START_CKPT}" \
      gene_num=${GENE_NUM} bin_num=${BIN_NUM} pos_embed=${USE_POSEMB} \
      dropout=${B_DROPOUT} l1_lambda=${B_L1} \
      epochs=${EPOCHS} batch_size=${BATCH} grad_acc=${GRAD_ACC} learning_rate=${LR} \
      cv_splits=${CV_SPLITS_B} use_bayes=true \
      kl_weight=${KL_WEIGHT} kl_anneal_steps=${KL_STEPS} bayes_mc_samples_train=${MC_SAMPLES} bayes_mc_samples_eval=${MC_SAMPLES_EVAL} pre_warm_epochs=${PRE_WARM} \
      weight_decay=${WEIGHT_DECAY} head_lr_mult=${HEAD_LR_MULT} patience=${EARLY_STOP_PATIENCE} \
      prune_target=${PRUNE_SECONDARY_TARGET} prune_step=${PRUNE_SECONDARY_STEP} prune_ft_epochs=${PRUNE_SECONDARY_FT_EPOCHS} \
      prune_f1_tol=${PRUNE_SECONDARY_F1_TOL} prune_dim=${PRUNE_DIM} prune_layers="${PRUNE_SECONDARY_LAYERS}"

    B_ARGS_R2=(
      --data_path "${DATA}"
      --model_path "${START_CKPT}"
      --ckpt_dir "${OUT_B}"
      --model_name "${B_MODEL_NAME_R2}"
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
    if [[ "${LAUNCH_CMD[*]}" == *"torch.distributed.launch"* ]]; then
      B_ARGS_R2=(--local_rank 0 "${B_ARGS_R2[@]}")
    fi

    "${LAUNCH_CMD[@]}" finetune_sparse_stageB.py "${B_ARGS_R2[@]}" 2>&1 | tee -- "${B_LOG_R2}"

    # 更新最终使用的 B model name/log 以供后续汇总
    B_MODEL_NAME="${B_MODEL_NAME_R2}"
    B_LOG="${B_LOG_R2}"
  else
    echo "[WARN] Round2 条件不满足，已跳过二轮剪枝。"
  fi
fi

# Run merged round (Bayes + pruning) if requested
if [[ "${MERGE_ROUNDS}" == "1" ]]; then
  echo ">>> [B] MERGE_ROUNDS=1 -> running a single combined Stage-B (Bayes + pruning)"
  # choose start ckpt: prefer HARDCODE, else A_BEST
  START_CKPT=""
  if [[ -n "${HARDCODE_ROUND2_CKPT:-}" && -f "${HARDCODE_ROUND2_CKPT}" ]]; then
    START_CKPT="${HARDCODE_ROUND2_CKPT}"
    echo "[INFO] Using HARDCODE_ROUND2_CKPT as start ckpt: ${START_CKPT}"
  elif [[ -f "${A_BEST}" ]]; then
    START_CKPT="${A_BEST}"
    echo "[INFO] Using A_BEST as start ckpt: ${START_CKPT}"
  else
    echo "[ERR] No valid start checkpoint found for merged run (HARDCODE_ROUND2_CKPT or A_BEST). Aborting merged run." >&2
    START_CKPT=""
  fi
  if [[ -n "${START_CKPT}" && -f "${START_CKPT}" ]]; then
    B_TAG_MERGE="B_merged_bayes_prune_do${B_DROPOUT}_l1${B_L1}_t${PRUNE_SECONDARY_TARGET}_s${PRUNE_SECONDARY_STEP}_ft${PRUNE_SECONDARY_FT_EPOCHS}_d${PRUNE_DIM}_L$(echo ${PRUNE_SECONDARY_LAYERS} | tr ',' '-')_kw${KL_WEIGHT}_ks${KL_STEPS}_mc${MC_SAMPLES}_pw${PRE_WARM}"
    B_MODEL_NAME_MERGE="stageB_${B_TAG_MERGE}"
    B_LOG_MERGE="${OUT_B}/${B_MODEL_NAME_MERGE}.log"

    save_json "${OUT_B}/stageB_merged_config.json" \
      stage="B_merged" data_path="${DATA}" init_ckpt="${START_CKPT}" \
      gene_num=${GENE_NUM} bin_num=${BIN_NUM} pos_embed=${USE_POSEMB} \
      dropout=${B_DROPOUT} l1_lambda=${B_L1} \
      epochs=${EPOCHS} batch_size=${BATCH} grad_acc=${GRAD_ACC} learning_rate=${LR} \
      cv_splits=${CV_SPLITS_B} use_bayes=true \
      kl_weight=${KL_WEIGHT} kl_anneal_steps=${KL_STEPS} bayes_mc_samples_train=${MC_SAMPLES} bayes_mc_samples_eval=${MC_SAMPLES_EVAL} pre_warm_epochs=${PRE_WARM} \
      weight_decay=${WEIGHT_DECAY} head_lr_mult=${HEAD_LR_MULT} patience=${EARLY_STOP_PATIENCE} \
      prune_target=${PRUNE_SECONDARY_TARGET} prune_step=${PRUNE_SECONDARY_STEP} prune_ft_epochs=${PRUNE_SECONDARY_FT_EPOCHS} \
      prune_f1_tol=${PRUNE_SECONDARY_F1_TOL} prune_dim=${PRUNE_DIM} prune_layers="${PRUNE_SECONDARY_LAYERS}"

    B_ARGS_MERGE=(
      --data_path "${DATA}"
      --model_path "${START_CKPT}"
      --ckpt_dir "${OUT_B}"
      --model_name "${B_MODEL_NAME_MERGE}"
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
      B_ARGS_MERGE+=(--bayes_head --kl_weight "${KL_WEIGHT}" --kl_anneal_steps "${KL_STEPS}" --bayes_mc_samples "${MC_SAMPLES_EVAL}")
    fi
    if [[ "${LAUNCH_CMD[*]}" == *"torch.distributed.launch"* ]]; then
      B_ARGS_MERGE=(--local_rank 0 "${B_ARGS_MERGE[@]}")
    fi

    "${LAUNCH_CMD[@]}" finetune_sparse_stageB.py "${B_ARGS_MERGE[@]}" 2>&1 | tee -- "${B_LOG_MERGE}"

    # 更新最终使用的 B model name/log 以供后续汇总
    B_MODEL_NAME="${B_MODEL_NAME_MERGE}"
    B_LOG="${B_LOG_MERGE}"
  else
    echo "[WARN] 无可用起始 ckpt，已跳过合并运行。" >&2
  fi

  # 因为合并运行已经替代了 round1/round2，跳过后续两轮逻辑
  # 跳转到 CSV 合并部分（通过继续脚本自然到达）
fi
############################################
# 汇总：把所有折CSV合并并做总表
############################################
echo ">>> Merging per-fold CSVs ..."

# 1) 合并A阶段：所有 *_metrics.csv -> A_all_folds_metrics.csv
merge_csv "${OUT_A}/${A_MODEL_NAME}[0-9]_metrics.csv" "${OUT_A}/A_all_folds_metrics.csv"

# 2) 合并B阶段：所有 *_metrics.csv -> B_all_folds_metrics.csv
merge_csv "${OUT_B}/${B_MODEL_NAME}[0-9]_metrics.csv" "${OUT_B}/B_all_folds_metrics.csv"

# 3) 从日志抽取关键行到 summary（保留追踪）
A_SUM="${OUT_A}/A_summary.csv"
echo 'line_no,val_f1,source' > "${A_SUM}"
if [[ -f "${A_LOG}" ]]; then
  # Use Python to extract "F1 Score:" lines robustly (avoids awk portability issues)
  python - <<'PY' "${A_LOG}" "${A_SUM}"
import re,sys
log=sys.argv[1]
out=sys.argv[2]
pat=re.compile(r'F1 Score:\s*([0-9.]+)')
with open(log,'r',errors='ignore') as f, open(out,'a') as o:
    for i,l in enumerate(f,1):
        m=pat.search(l)
        if m:
            o.write(f"{i},{m.group(1)},A_val\n")
print(f"[OK] wrote {out}")
PY
else
  echo "[WARN] A log not found: ${A_LOG} — skipping A summary extraction" >&2
fi

B_SUM="${OUT_B}/B_summary.csv"
echo 'line_no,val_f1,source' > "${B_SUM}"
if [[ -f "${B_LOG}" ]]; then
  # Extract F1 Score lines, Post-Prune F1, and Prune Done best_F1 lines using Python (portable)
  python - <<'PY' "${B_LOG}" "${B_SUM}"
import re,sys
log=sys.argv[1]
out=sys.argv[2]
pat_val=re.compile(r'F1 Score:\s*([0-9.]+)')
pat_post=re.compile(r'Post-Prune.*Val F1=\s*([0-9.]+)')
pat_prune_done=re.compile(r'Prune Done.*best_F1=\s*([0-9.]+)')
with open(log,'r',errors='ignore') as f, open(out,'a') as o:
    for i,l in enumerate(f,1):
        m=pat_val.search(l)
        if m:
            o.write(f"{i},{m.group(1)},B_val\n")
        m2=pat_post.search(l)
        if m2:
            o.write(f"{i},{m2.group(1)},B_post_prune_ft\n")
        m3=pat_prune_done.search(l)
        if m3:
            o.write(f"{i},{m3.group(1)},B_prune_done_best\n")
print(f"[OK] wrote {out}")
PY
else
  echo "[WARN] B log not found: ${B_LOG} — skipping B summary extraction" >&2
fi
if [ -s "${B_SUM}" ]; then
  sort -t, -k1,1n "${B_SUM}" | cut -d, -f2- > "${B_SUM}.tmp" && mv "${B_SUM}.tmp" "${B_SUM}"
fi

# 4) 计算每阶段“各折最后一次验证”的均值（Acc, F1）到一个总汇总
TOTAL="${OUTROOT}/overall_summary.csv"
echo "stage,folds,last_epoch_avg_acc,last_epoch_avg_f1" > "${TOTAL}"

# 从“每折 CSV 的通配符模式”计算各折最后一行的 acc/f1 均值
calc_last_avg () {
  # $1: per-fold csv 的 glob 模式，例如：/path/stageA_A_det...*[0-9]_metrics.csv
  local pattern="$1"
  local files=( ${pattern} )
  local acc_sum=0
  local f1_sum=0
  local n=0
  if [[ ${#files[@]} -eq 0 ]]; then
    echo ","
    return
  fi
  for f in "${files[@]}"; do
    [[ -s "$f" ]] || continue
    # 从上到下扫描，保留“最近一行同时具有数值型 val_acc(第5列) 和 f1(第6列)”的行
    local last_valid
    last_valid=$(awk -F, '($5 ~ /^[0-9.]+$/ && $6 ~ /^[0-9.]+$/){line=$0} END{print line}' "$f")
    if [[ -n "$last_valid" ]]; then
      local acc=$(echo "$last_valid" | awk -F, '{print $(5)}')
      local f1=$(echo  "$last_valid" | awk -F, '{print $(6)}')
      acc_sum=$(awk -v a="$acc_sum" -v b="$acc" 'BEGIN{printf "%.6f", a + b}')
      f1_sum=$(awk -v a="$f1_sum" -v b="$f1" 'BEGIN{printf "%.6f", a + b}')
      n=$((n+1))
    fi
  done
  if [[ $n -gt 0 ]]; then
    awk -v a="$acc_sum" -v b="$f1_sum" -v n="$n" 'BEGIN{printf "%.6f,%.6f", a/n, b/n}'
  else
    echo ","
  fi
}

# 原先是传入合并文件名，这里改为直接传入每折 CSV 的 glob 模式
A_AVG=$(calc_last_avg "${OUT_A}/${A_MODEL_NAME}[0-9]_metrics.csv")
B_AVG=$(calc_last_avg "${OUT_B}/${B_MODEL_NAME}[0-9]_metrics.csv")

echo "A,${CV_SPLITS_A},${A_AVG}" >> "${TOTAL}"
echo "B,${CV_SPLITS_B},${B_AVG}" >> "${TOTAL}"

echo "[OK] CSV 已生成："
echo " - ${OUT_A}/A_all_folds_metrics.csv"
echo " - ${OUT_B}/B_all_folds_metrics.csv"
echo " - ${OUTROOT}/overall_summary.csv"
echo " - ${OUT_A}/stageA_config.json"
echo " - ${OUT_B}/stageB_config.json"
echo " - ${OUT_A}/A_summary.csv"
echo " - ${OUT_B}/B_summary.csv"

echo "<<< 完成 A→B 全流程"

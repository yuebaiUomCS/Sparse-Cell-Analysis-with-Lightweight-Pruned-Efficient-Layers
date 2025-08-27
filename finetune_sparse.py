# -*- coding: utf-8 -*-
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import random
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.utils.prune as prune
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import scanpy as sc
import pickle as pkl

from performer_pytorch import PerformerLM
from utils import *  # 需包含我们改过的 get_reduced / distributed_concat / save_best_ckpt / CosineAnnealingWarmupRestarts

# -------------------------
# Tee 日志重定向
# -------------------------
import sys
import time
class Tee(object):
    def __init__(self, name, mode='w'):
        self.file = open(name, mode)
        self.stdout = sys.__stdout__
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()

# Add utility and reproducibility flags near top (after imports)
import tempfile, shutil
import gc
import torch

# Reproducibility / cuDNN options (choose deterministic or faster)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# If you prefer speed uncomment below instead
# torch.backends.cudnn.benchmark = True

# Atomic append helper for metrics/CSV writing (must be defined before first use)
def _atomic_append_line(path, line: str):
    """Append a single line to path atomically to avoid races/corruption.
    Creates parent dir if missing.
    """
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d or '.')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(line.rstrip('\n') + '\n')
        # append to target file (create if missing)
        with open(path, 'a') as out:
            with open(tmp, 'r') as _in:
                shutil.copyfileobj(_in, out)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

# Pruning helper that handles BayesianLinear by pruning weight_mu directly
def _prune_l2_structured_on_effective_weight(layer, amount, dim):
    try:
        from torch.nn.utils import prune
    except Exception:
        prune = None
    if prune is None:
        return
    if hasattr(layer, '__class__') and layer.__class__.__name__.endswith('BayesianLinear'):
        # prune the learned mean parameter directly
        if hasattr(layer, 'weight_mu'):
            prune.ln_structured(layer, name='weight_mu', amount=amount, n=2, dim=dim)
            prune.remove(layer, 'weight_mu')
    else:
        if hasattr(layer, 'weight'):
            prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=dim)
            prune.remove(layer, 'weight')

# -------------------------
# 解析参数
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=1, help='Batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Validate every N epochs.')
parser.add_argument("--patience", type=int, default=10, help='Early stopping patience in epochs (trigger > patience stops).')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Finetune data path.')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth', help='Pretrained/Phase-A ckpt path.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Checkpoint dir.')
parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')
parser.add_argument("--l1_lambda", type=float, default=0.0, help='L1 weight.')

# structured prune 超参
parser.add_argument("--prune_target", type=float, default=0.4, help="total structured prune ratio (0~1)")
parser.add_argument("--prune_step", type=float, default=0.1, help="step ratio per prune round (0~1)")
parser.add_argument("--prune_ft_epochs", type=int, default=3, help="finetune epochs after each prune step")
parser.add_argument("--prune_f1_tol", type=float, default=0.01, help="allowed F1 drop per round before stopping")
parser.add_argument("--prune_dim", type=int, default=0, choices=[0,1], help="0=prune output neurons (rows), 1=input features (cols)")
parser.add_argument("--prune_layers", type=str, default="fc1", help="layers under to_out, e.g. 'fc1' or 'fc1,fc2'")

# pos embed flag
parser.add_argument("--pos_embed", action="store_true", help="Enable Gene2vec/G2V positional embedding")
parser.add_argument("--no_pos_embed", dest="pos_embed", action="store_false")
parser.set_defaults(pos_embed=True)

# Bayes 头 & KL
parser.add_argument("--bayes_head", action="store_true", help="use BayesianLinear for fc1/fc2 in to_out head")
parser.add_argument("--kl_weight", type=float, default=1e-5, help="global weight for KL term")
parser.add_argument("--kl_anneal_steps", type=int, default=20000, help="linear warmup steps for KL from 0->kl_weight")
parser.add_argument("--bayes_mc_samples", type=int, default=8, help="MC samples in eval; 1 for speed")
parser.add_argument("--cv_splits", type=int, default=5, help="Number of CV folds; set 1 for a single split")
parser.add_argument("--pre_warm_epochs", type=int, default=2, help="Number of initial deterministic warmup epochs for Bayes head")
parser.add_argument("--head_lr_mult", type=float, default=0.5, help="Multiplier for head learning rate relative to base LR")
# 额外
parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay for optimizer")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for classifier head")

args = parser.parse_args()
head_lr_mult = getattr(args, 'head_lr_mult', 0.5)
# Map CLI patience into runtime variable
# PATIENCE controls early stopping: if validation metric doesn't improve for >PATIENCE epochs, stop
# Note: finetune loop uses trigger_times > PATIENCE to break
PATIENCE = args.patience

# -------------------------
# 读取关键参数
# -------------------------
prune_target    = args.prune_target
prune_step      = args.prune_step
prune_ft_epochs = args.prune_ft_epochs
prune_f1_tol    = args.prune_f1_tol
prune_dim       = args.prune_dim
prune_layers    = tuple([s.strip() for s in args.prune_layers.split(",") if s.strip()])

l1_lambda       = args.l1_lambda
use_bayes_head  = args.bayes_head
kl_weight       = args.kl_weight
kl_anneal_steps = args.kl_anneal_steps
original_kl_anneal = kl_anneal_steps
bayes_mc_samples= args.bayes_mc_samples
pre_warm_epochs = args.pre_warm_epochs

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

model_name = args.model_name
modelraw = model_name
ckpt_dir = args.ckpt_dir
os.makedirs(ckpt_dir, exist_ok=True)

# -------------------------
# DDP/设备初始化（兼容 torchrun / 直接 python）
# -------------------------
def _intenv(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

WORLD_SIZE = _intenv("WORLD_SIZE", 1)
RANK       = _intenv("RANK", 0)
LOCAL_RANK = _intenv("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0)

ddp = WORLD_SIZE > 1
use_cuda = torch.cuda.is_available()

if ddp and not dist.is_initialized():
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

# 设备绑定
if use_cuda:
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f"cuda:{LOCAL_RANK}")
else:
    device = torch.device("cpu")

# 进程信息 & 主进程判断
world_size = dist.get_world_size() if dist.is_initialized() else 1
rank       = dist.get_rank() if dist.is_initialized() else 0
is_master  = (rank == 0)

# 设种子：不同进程不同
seed_all(SEED + rank)

# -------------------------
# 数据集
# -------------------------
class SCDataset(Dataset):
    """随机抽样行（与 baseline 对齐）。可选顺序访问用于验证。"""
    def __init__(self, data, label, random: bool = True):
        super().__init__()
        self.data = data
        self.label = label
        self.random = bool(random)

    def __getitem__(self, index):
        if self.random:
            rand_idx = random.randint(0, self.data.shape[0] - 1)
        else:
            rand_idx = index
        row = self.data[rand_idx]
        if hasattr(row, 'toarray') and not isinstance(row, np.ndarray):
            arr = row.toarray().ravel()
        else:
            arr = np.array(row).ravel()
        arr[arr > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(arr).long()
        full_seq = torch.cat((full_seq, torch.tensor([0], dtype=torch.long))).to(device)
        seq_label = self.label[rand_idx]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]

# -------------------------
# Bayes 线性层
# -------------------------
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.02))
        self.weight_logvar = nn.Parameter(torch.empty(out_features, in_features).fill_(-5.0))
        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_logvar = nn.Parameter(torch.ones(out_features) * -5.0)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_logvar', None)
        self.bias = bias
        self._force_deterministic = False

    def forward(self, x, sample=False):
        try:
            self.weight_logvar.data.clamp_(-10.0, 2.0)
            if self.bias:
                self.bias_logvar.data.clamp_(-10.0, 2.0)
        except Exception:
            pass
        do_sample = (self.training or sample) and (not getattr(self, '_force_deterministic', False))
        if do_sample:
            eps_w = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * eps_w
            if self.bias:
                eps_b = torch.randn_like(self.bias_mu)
                bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * eps_b
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.bias else None
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        kl_w = 0.5 * torch.sum(self.weight_mu.pow(2) + torch.exp(self.weight_logvar) - self.weight_logvar - 1.0)
        if self.bias:
            kl_b = 0.5 * torch.sum(self.bias_mu.pow(2) + torch.exp(self.bias_logvar) - self.bias_logvar - 1.0)
        else:
            kl_b = torch.tensor(0.0, device=self.weight_mu.device)
        return kl_w + kl_b

# -------------------------
# 分类头（支持 Bayes）
# -------------------------
class Identity(nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10, use_bayes=False):
        super().__init__()
        self.use_bayes = use_bayes
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self._bayes_eval_sample = False
        if self.use_bayes:
            self.fc1 = BayesianLinear(SEQ_LEN, 512, bias=True)
            self.fc2 = BayesianLinear(512, h_dim, bias=True)
        else:
            self.fc1 = nn.Linear(SEQ_LEN, 512, bias=True)
            self.fc2 = nn.Linear(512, h_dim, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(h_dim, out_dim, bias=True)

    def enable_bayes_sampling(self, flag: bool = True):
        self._bayes_eval_sample = bool(flag)

    def set_force_deterministic(self, flag: bool = True):
        if not self.use_bayes:
            return
        for name in ('fc1', 'fc2'):
            lyr = getattr(self, name, None)
            if isinstance(lyr, BayesianLinear):
                lyr._force_deterministic = bool(flag)

    def forward(self, x, sample=False):
        x = x[:, None, :, :]
        x = self.conv1(x); x = self.act(x)
        x = x.view(x.shape[0], -1)
        sample = sample or getattr(self, "_bayes_eval_sample", False)
        if self.use_bayes:
            x = self.fc1(x, sample=sample)
        else:
            x = self.fc1(x)
        x = self.act1(x); x = self.dropout1(x)
        if self.use_bayes:
            x = self.fc2(x, sample=sample)
        else:
            x = self.fc2(x)
        x = self.act2(x); x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def kl(self):
        if not self.use_bayes:
            return torch.tensor(0.0, device=self.fc3.weight.device)
        return self.fc1.kl_divergence() + self.fc2.kl_divergence()

# -------------------------
# 工具函数
# -------------------------
def get_model_head(model):
    return model.module.to_out if isinstance(model, DDP) else model.to_out

# -------------------------
# Helpers to port deterministic head -> Bayesian head
# -------------------------
def _strip_module_prefix(state_dict, prefix='module.'):
    """Remove common DataParallel/DistributedDataParallel 'module.' prefix from state-dict keys."""
    new = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new[k[len(prefix):]] = v
        else:
            new[k] = v
    return new


def port_det_head_to_bayes(
    model,
    ckpt_state_dict,
    head_attr='to_out',
    layer_names=('fc1','fc2','fc3'),
    unset_logvar=-9.0,
    is_master=True
):
    """Robustly copy deterministic head params from a checkpoint state-dict into the current model's head.

    Behavior improvements:
    - Accepts various key naming conventions (with/without prefixes, weight vs weight_mu in ckpt)
    - Prefers keys whose tensor shapes match target layer shapes
    - Copies existing logvar keys from ckpt if present; otherwise sets logvar to unset_logvar
    - Provides detailed diagnostics for missing keys / shape mismatches
    """
    sd = ckpt_state_dict
    if isinstance(sd, dict) and ('model_state_dict' in sd or 'state_dict' in sd):
        sd = sd.get('model_state_dict', sd.get('state_dict', sd))
    if not isinstance(sd, dict):
        return False, {'error': 'invalid_state_dict'}
    sd = _strip_module_prefix(sd)

    head = getattr(model, head_attr, None)
    if head is None:
        if is_master:
            print(f"[WARN] Model has no '{head_attr}' to port into.", flush=True)
        return False, {'error': 'no_head'}

    success = False
    diagnostics = {'copied': [], 'shape_mismatch': [], 'missing': [], 'errors': [], 'candidates': {}}

    def _get_shape(v):
        try:
            return tuple(v.shape)
        except Exception:
            try:
                return tuple(torch.tensor(v).shape)
            except Exception:
                return None

    # helper to convert checkpoint value to torch tensor on CPU for copying
    def _to_tensor(v):
        if isinstance(v, torch.Tensor):
            return v
        try:
            return torch.tensor(v)
        except Exception:
            return None

    for name in layer_names:
        lyr = getattr(head, name, None)
        if lyr is None:
            if is_master:
                print(f"[WARN] Head missing layer '{name}', skip.", flush=True)
            diagnostics['missing'].append(name)
            continue

        # collect candidate keys that reference this layer name
        candidates = [k for k in sd.keys() if f".{name}." in k or k.endswith(f"{name}.weight") or k.endswith(f"{name}.bias")]
        diagnostics['candidates'][name] = candidates

        # determine target shapes
        target_w_shape = None
        target_b_shape = None
        if hasattr(lyr, 'weight_mu'):
            target_w_shape = tuple(lyr.weight_mu.shape)
        elif hasattr(lyr, 'weight'):
            target_w_shape = tuple(lyr.weight.shape)
        if hasattr(lyr, 'bias_mu') and lyr.bias_mu is not None:
            target_b_shape = tuple(lyr.bias_mu.shape)
        elif hasattr(lyr, 'bias') and lyr.bias is not None:
            target_b_shape = tuple(lyr.bias.shape)

        # find best matching weight key in checkpoint
        weight_key = None
        bias_key = None
        weight_logvar_key = None
        bias_logvar_key = None

        # prioritized key names to look for
        pref_weight_names = [f"{name}.weight_mu", f"{name}.weight", f"{name}.weight_mean"]
        pref_bias_names   = [f"{name}.bias_mu", f"{name}.bias", f"{name}.bias_mean"]
        pref_wlog_names   = [f"{name}.weight_logvar", f"{name}.weight_var", f"{name}.weight_sigma2"]
        pref_blog_names   = [f"{name}.bias_logvar", f"{name}.bias_var"]

        # search exact-suffix matches first
        for k in sd.keys():
            for pk in pref_weight_names:
                if k.endswith(pk):
                    weight_key = k; break
            for pb in pref_bias_names:
                if k.endswith(pb):
                    bias_key = k; break
            for pw in pref_wlog_names:
                if k.endswith(pw):
                    weight_logvar_key = k; break
            for pbv in pref_blog_names:
                if k.endswith(pbv):
                    bias_logvar_key = k; break
            if weight_key and bias_key and weight_logvar_key and bias_logvar_key:
                break

        # if not found, pick candidates by shape match
        if weight_key is None and candidates:
            for k in candidates:
                if k.split('.')[-1] in ('weight','weight_mu'):
                    sh = _get_shape(sd[k])
                    if sh is not None and target_w_shape is not None and sh == target_w_shape:
                        weight_key = k; break
            # fallback to first weight-like candidate
            if weight_key is None:
                for k in candidates:
                    if k.split('.')[-1] in ('weight','weight_mu'):
                        weight_key = k; break

        if bias_key is None and candidates:
            for k in candidates:
                if k.split('.')[-1] in ('bias','bias_mu'):
                    sh = _get_shape(sd[k])
                    if sh is not None and target_b_shape is not None and sh == target_b_shape:
                        bias_key = k; break
            if bias_key is None:
                for k in candidates:
                    if k.split('.')[-1] in ('bias','bias_mu'):
                        bias_key = k; break

        # try find logvar keys similarly
        if weight_logvar_key is None and candidates:
            for k in candidates:
                if k.split('.')[-1] in ('weight_logvar','weight_var','weight_sigma2'):
                    weight_logvar_key = k; break
        if bias_logvar_key is None and candidates:
            for k in candidates:
                if k.split('.')[-1] in ('bias_logvar','bias_var'):
                    bias_logvar_key = k; break

        try:
            # copy weight
            if weight_key and weight_key in sd:
                wval = sd[weight_key]
                wt = _to_tensor(wval)
                if wt is not None:
                    if hasattr(lyr, 'weight_mu') and tuple(wt.shape) == target_w_shape:
                        with torch.no_grad():
                            lyr.weight_mu.copy_(wt)
                        diagnostics['copied'].append((name, 'weight_mu', tuple(wt.shape)))
                        success = True
                    elif hasattr(lyr, 'weight') and tuple(wt.shape) == target_w_shape:
                        with torch.no_grad():
                            lyr.weight.copy_(wt)
                        diagnostics['copied'].append((name, 'weight', tuple(wt.shape)))
                        success = True
                    else:
                        diagnostics['shape_mismatch'].append((name, 'weight', tuple(wt.shape), target_w_shape))
                else:
                    diagnostics['errors'].append(f"Unable to convert checkpoint weight for {name}")
            else:
                diagnostics['missing'].append((name, 'weight'))

            # copy bias
            if bias_key and bias_key in sd:
                bval = sd[bias_key]
                bt = _to_tensor(bval)
                if bt is not None:
                    if hasattr(lyr, 'bias_mu') and lyr.bias_mu is not None and tuple(bt.shape) == target_b_shape:
                        with torch.no_grad():
                            lyr.bias_mu.copy_(bt)
                        diagnostics['copied'].append((name, 'bias_mu', tuple(bt.shape)))
                    elif hasattr(lyr, 'bias') and lyr.bias is not None and tuple(bt.shape) == target_b_shape:
                        with torch.no_grad():
                            lyr.bias.copy_(bt)
                        diagnostics['copied'].append((name, 'bias', tuple(bt.shape)))
                    else:
                        diagnostics['shape_mismatch'].append((name, 'bias', tuple(bt.shape), target_b_shape))
                else:
                    diagnostics['errors'].append(f"Unable to convert checkpoint bias for {name}")
            else:
                diagnostics['missing'].append((name, 'bias'))

            # copy/set logvars
            # weight logvar
            if hasattr(lyr, 'weight_logvar'):
                if weight_logvar_key and weight_logvar_key in sd:
                    wlv = _to_tensor(sd[weight_logvar_key])
                    if wlv is not None and tuple(wlv.shape) == tuple(lyr.weight_logvar.shape):
                        with torch.no_grad():
                            lyr.weight_logvar.copy_(wlv)
                        diagnostics['copied'].append((name, 'weight_logvar_from_ckpt', tuple(wlv.shape)))
                    else:
                        with torch.no_grad():
                            lyr.weight_logvar.data.fill_(unset_logvar)
                        diagnostics['copied'].append((name, 'weight_logvar_set', unset_logvar))
                else:
                    with torch.no_grad():
                        lyr.weight_logvar.data.fill_(unset_logvar)
                    diagnostics['copied'].append((name, 'weight_logvar_set', unset_logvar))

            # bias logvar
            if hasattr(lyr, 'bias_logvar') and getattr(lyr, 'bias', True):
                if bias_logvar_key and bias_logvar_key in sd:
                    blv = _to_tensor(sd[bias_logvar_key])
                    if blv is not None and tuple(blv.shape) == tuple(lyr.bias_logvar.shape):
                        with torch.no_grad():
                            lyr.bias_logvar.copy_(blv)
                        diagnostics['copied'].append((name, 'bias_logvar_from_ckpt', tuple(blv.shape)))
                    else:
                        with torch.no_grad():
                            lyr.bias_logvar.data.fill_(unset_logvar)
                        diagnostics['copied'].append((name, 'bias_logvar_set', unset_logvar))
                else:
                    with torch.no_grad():
                        lyr.bias_logvar.data.fill_(unset_logvar)
                    diagnostics['copied'].append((name, 'bias_logvar_set', unset_logvar))

        except Exception as e:
            diagnostics['errors'].append(str(e))
            if is_master:
                print(f"[WARN] Failed to port layer {name}: {e}", flush=True)

    if is_master:
        if diagnostics['copied']:
            print(f"[INFO] Ported deterministic head weights into Bayes head (details):", flush=True)
            for it in diagnostics['copied']:
                print(f"  COPIED: {it}", flush=True)
        if diagnostics['shape_mismatch']:
            print(f"[INFO] Shape mismatches during porting:", flush=True)
            for it in diagnostics['shape_mismatch']:
                print(f"  MISMATCH: {it}", flush=True)
        if diagnostics['missing']:
            print(f"[INFO] Missing keys or layers: {diagnostics['missing']}", flush=True)
        if diagnostics['errors']:
            print(f"[INFO] Errors during porting: {diagnostics['errors']}", flush=True)
        # also print candidates map to aid debugging
        for nm, cand in diagnostics['candidates'].items():
            if cand:
                print(f"[DEBUG] Candidate keys for {nm}: {cand}", flush=True)

    return success, diagnostics



def try_port_ckpt_head_to_bayes(model, ckpt, is_master=True):
    """Wrapper: accept full ckpt object, extract state-dict and call port_det_head_to_bayes."""
    if ckpt is None:
        return False
    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    return port_det_head_to_bayes(model, sd, is_master=is_master)

def get_sparsity(module):
    num_zero, num_total = 0, 0
    for _, param in module.named_parameters():
        num_zero += (param.abs() < 1e-6).sum().item()
        num_total += param.numel()
    return num_zero / num_total if num_total > 0 else 0.0

def elem_sparsity(module, eps=1e-8):
    zero, total = 0, 0
    for _, p in module.named_parameters():
        if p is None or p.numel() == 0:
            continue
        z = (p.abs() < eps).sum().item()
        zero += z
        total += p.numel()
    return (zero / total) if total > 0 else 0.0

def structured_sparsity_linear(layer, dim=0, eps=1e-8):
    W = getattr(layer, "weight_mu", None)
    if W is None:
        W = getattr(layer, "weight", None)
    if W is None:
        return 0.0
    W = W.detach()
    if dim == 0:
        keep = (W.abs() >= eps).any(dim=1)
        return 1.0 - (keep.float().mean().item())
    else:
        keep = (W.abs() >= eps).any(dim=0)
        return 1.0 - (keep.float().mean().item())

def forward_eval(model, x, mc=1):
    def _normalize_logits_shape(tensor):
        if tensor.dim() == 3:
            return tensor[:, 0, :]
        elif tensor.dim() == 2:
            return tensor
        elif tensor.dim() == 1:
            return tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected output shape in forward_eval: {tensor.shape}")

    out = _normalize_logits_shape(model(x))
    if mc <= 1:
        return out

    head = get_model_head(model)
    bayes_capable = getattr(head, 'use_bayes', False)
    if bayes_capable:
        try:
            head.enable_bayes_sampling(True)
        except Exception:
            pass

    logits_sum = torch.zeros_like(out)
    with torch.no_grad():
        for _ in range(mc):
            sample_out = _normalize_logits_shape(model(x))
            logits_sum = logits_sum + sample_out

    if bayes_capable:
        try:
            head.enable_bayes_sampling(False)
        except Exception:
            pass

    return logits_sum / mc

from sklearn.metrics import f1_score

def quick_eval(model, loader, device, world_size, val_sampler, mc=1, loss_fn=None):
    model.eval()
    running_loss = 0.0
    predictions, truths = [], []
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            logits = forward_eval(model, x, mc=mc)
            loss = loss_fn(logits, y)
            running_loss += loss.item()
            pred = logits.argmax(dim=-1)
            predictions.append(pred)
            truths.append(y)

    preds_tensor  = torch.cat(predictions, dim=0) if len(predictions)>0 else torch.tensor([], device=device, dtype=torch.long)
    truths_tensor = torch.cat(truths, dim=0) if len(truths)>0 else torch.tensor([], device=device, dtype=torch.long)
    total_size = len(loader.dataset)

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        predictions = distributed_concat(preds_tensor,  total_size, world_size)
        truths      = distributed_concat(truths_tensor, total_size, world_size)
    else:
        predictions = preds_tensor[:total_size]
        truths      = truths_tensor[:total_size]

    predictions = np.array(predictions.cpu()); truths = np.array(truths.cpu())
    acc = (predictions == truths).mean() if truths.size > 0 else 0.0
    f1 = f1_score(truths, predictions, average='macro') if truths.size > 0 else 0.0
    val_loss = running_loss / max(1, len(loader))
    return acc, f1, val_loss

def finetune_one_epoch(model, train_loader, optimizer, loss_fn, device,
                       current_epoch: int,
                       pre_warm_epochs: int,
                       use_bayes_head_flag: bool):
    """一个 epoch 训练（支持 Bayes 预热、KL 退火、梯度累积、DDP no_sync）"""
    model.train()
    accum_steps = max(1, GRADIENT_ACCUMULATION)
    optimizer.zero_grad()
    global global_step

    running_loss = 0.0
    cum_acc = 0.0
    batch_count = 0

    in_warmup = use_bayes_head_flag and (current_epoch <= max(0, int(pre_warm_epochs)))

    head = get_model_head(model)
    if use_bayes_head_flag:
        try:
            head.set_force_deterministic(in_warmup)
            head.enable_bayes_sampling(False)
        except Exception:
            pass

    for index, (data, labels) in enumerate(train_loader, start=1):
        batch_count = index
        data, labels = data.to(device), labels.to(device)

        # L1（Bayes 跳过 logvar）按参数量归一
        l1_reg = 0.0
        l1_n = 0
        for n, p in head.named_parameters():
            if 'logvar' in n or p is None or p.numel() == 0:
                continue
            l1_reg += p.abs().sum()
            l1_n += p.numel()
        if l1_n > 0:
            l1_reg = l1_reg / l1_n

        if use_bayes_head_flag:
            if in_warmup:
                kl_scale = 0.0
            else:
                kl_scale = min(1.0, global_step / max(1, kl_anneal_steps)) * kl_weight
        else:
            kl_scale = 0.0

        logits = model(data)
        loss = loss_fn(logits, labels)
        running_loss += loss.item()

        if l1_lambda > 0:
            # use provided l1_lambda for all heads (Bayes uses weight_mu in regularizer computation)
            loss = loss + l1_lambda * l1_reg
        if use_bayes_head_flag:
            kl = head.kl()
            loss = loss + kl_scale * kl

        loss = loss / accum_steps

        if ddp and ((index % accum_steps) != 0):
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

        if (index % accum_steps) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if use_bayes_head_flag:
                try:
                    for nm in ('fc1', 'fc2'):
                        lyr = getattr(head, nm, None)
                        if isinstance(lyr, BayesianLinear):
                            lyr.weight_logvar.data.clamp_(-10.0, 2.0)
                            if lyr.bias:
                                lyr.bias_logvar.data.clamp_(-10.0, 2.0)
                except Exception:
                    pass
            optimizer.zero_grad()
            global_step += 1

        final = logits.softmax(dim=-1).argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = (final == labels).sum()
        cum_acc += (correct_num.float() / pred_num).item()

    num_steps = max(1, batch_count)
    epoch_loss = running_loss / num_steps
    epoch_acc = 100.0 * (cum_acc / num_steps)
    return epoch_loss, epoch_acc

def progressive_structured_prune(
    model, train_loader, val_loader, optimizer, loss_fn, device,
    world_size, val_sampler,
    target=0.4, step=0.1, max_ft_epochs=3, f1_drop_tol=0.01,
    dim=0, layer_names=('fc1',), is_master=True, mc_eval=1
):
    
    head = get_model_head(model)

    pre_acc, pre_f1, _ = quick_eval(model, val_loader, device, world_size, val_sampler, mc=mc_eval, loss_fn=loss_fn)
    if is_master:
        print(f"[Pre-Prune] Acc={pre_acc:.4f}, F1={pre_f1:.4f}")
    best_f1 = pre_f1
    done = 0.0
    while done < target and step > 0:
        left = min(step, target - done)
        for name in layer_names:
            layer = getattr(head, name, None)
            if layer is None:
                raise ValueError(f"to_out 没有层 '{name}'，可用层有: {list(dict(head.named_children()).keys())}")
            _prune_l2_structured_on_effective_weight(layer, amount=left, dim=dim)

        post_acc0, post_f10, _ = quick_eval(model, val_loader, device, world_size, val_sampler, mc=mc_eval, loss_fn=loss_fn)
        if is_master:
            print(f"[Post-Prune(no ft)] +{done+left:.2f} Acc={post_acc0:.4f}, F1={post_f10:.4f}")

        old_lrs = [pg["lr"] for pg in optimizer.param_groups]
        for pg in optimizer.param_groups: pg["lr"] *= 0.5
        for _ in range(max_ft_epochs):
            finetune_one_epoch(model, train_loader, optimizer, loss_fn, device,
                               current_epoch=0, pre_warm_epochs=0, use_bayes_head_flag=use_bayes_head)
        for pg, old in zip(optimizer.param_groups, old_lrs): pg["lr"] = old

        post_acc1, post_f11, _ = quick_eval(model, val_loader, device, world_size, val_sampler, mc=mc_eval, loss_fn=loss_fn)
        if is_master:
            print(f"[Post-Prune(ft)]   +{done+left:.2f} Acc={post_acc1:.4f}, F1={post_f11:.4f}")

        if post_f11 + 1e-12 < (best_f1 - f1_drop_tol):
            if is_master:
                print(f"[Stop] F1 drop exceeds tolerance ({f1_drop_tol:.3f}). Stopping prune.")
            break
        else:
            best_f1 = max(best_f1, post_f11)
            done += left
    if is_master:
        print(f"[Prune Done] total structured prune ~= {done:.2f}, best_F1={best_f1:.4f}")

    # Return richer info to allow comparative summaries
    return {'best_f1': best_f1, 'pre_acc': pre_acc, 'pre_f1': pre_f1, 'final_pruned': done}

# -------------------------
# 读取数据
# -------------------------
print("[INFO] Loading data...")
dataraw = sc.read_h5ad(args.data_path)
label_dict, label = np.unique(np.array(dataraw.obs['cell_type']), return_inverse=True)
print("[INFO] Data loaded successfully.")

with open('label_dict', 'wb') as fp:
    pkl.dump(label_dict, fp)
with open('label', 'wb') as fp:
    pkl.dump(label, fp)
class_num = np.unique(label, return_counts=True)[1].tolist()
label = torch.from_numpy(label)
dataraw = dataraw.X

print(f"[DEBUG] Data shape: {dataraw.shape}, Label shape: {label.shape}", flush=True)

number_split = 0
sss = StratifiedShuffleSplit(n_splits=args.cv_splits, test_size=0.2, random_state=SEED)

global_step = 0
start_time = time.time()

for index_train, index_val in sss.split(dataraw, label):
    global_step = 0  # <<< reset per-fold
    number_split += 1
    print(f"[INFO] Starting fold {number_split}...", flush=True)

    cur_model_name = modelraw + str(number_split)

    log_file = os.path.join(ckpt_dir, f"{cur_model_name}.log")
    if is_master:
        tee = Tee(log_file); sys.stdout = tee; sys.stderr = tee
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(message)s', filemode='w')
    if is_master:
        print(f"Logging results to: {log_file}", flush=True)
        logging.info(f"===== Fold {number_split} Start =====")

    metrics_file = os.path.join(ckpt_dir, f"{cur_model_name}_metrics.csv")
    if is_master and not os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'w') as mf:
                # Updated header: keep val_acc & f1 as columns 5/6; append kl_term,kl_scale before GPU/memory/time
                mf.write('epoch,train_loss,train_acc,val_loss,val_acc,f1,post_prune_ft_f1,prune_done_best_f1,total_params,nonzero_params,to_out_elem_sparsity,to_out_struct_sparsity,global_sparsity,kl_term,kl_scale,gpu_usage,memory_usage,compute_time\n')
        except Exception:
            pass

    data_train, label_train = dataraw[index_train], label[index_train]
    data_val, label_val = dataraw[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train, random=True)
    val_dataset = SCDataset(data_val, label_val, random=False)

    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False)
    else:
        from torch.utils.data import RandomSampler, SequentialSampler
        train_sampler = RandomSampler(train_dataset)
        val_sampler   = SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader  = DataLoader(val_dataset,   batch_size=BATCH_SIZE, sampler=val_sampler)

    # 自动计算 kl_anneal_steps: 基于每 epoch 的 optimizer 更新数（考虑梯度累积）
    try:
        updates_per_epoch = math.ceil(len(train_loader) / max(1, GRADIENT_ACCUMULATION))
        total_updates = updates_per_epoch * max(1, EPOCHS)
        # 推荐使用总更新数的 20% 作为 anneal window
        auto_kl = max(1, int(0.2 * total_updates))
        # 覆盖条件：用户显式传入 <=0 或传入的大于实际总步数（表明未调整过）
        if (original_kl_anneal is None) or (original_kl_anneal <= 0) or (original_kl_anneal > total_updates):
            kl_anneal_steps = auto_kl
            if is_master:
                print(f"[AUTO] kl_anneal_steps auto-set -> {kl_anneal_steps} (updates_per_epoch={updates_per_epoch}, total_updates={total_updates})", flush=True)
        else:
            if is_master:
                print(f"[INFO] using provided kl_anneal_steps={original_kl_anneal} (total_updates={total_updates})", flush=True)
    except Exception as e:
        if is_master:
            print(f"[WARN] Failed to auto-compute kl_anneal_steps: {e}", flush=True)

    # ---- 构建与加载骨干 ----
    model = PerformerLM(
        num_tokens = (args.bin_num + 2),
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = POS_EMBED_USING
    )
    # Hardcoded ckpt override (per request)
    HARDCODED_CKPT = "/linux/scBERT/final_ab_run/stageBstageB_B_round1_bayes_do0.5_l10_t0.0_s0.05_ft4_d0_Lfc1_kw1e-6_ks10000_mc1_pw61_best.pth"
    if os.path.exists(HARDCODED_CKPT):
        use_ckpt_path = HARDCODED_CKPT
        if is_master:
            print(f"[FORCE] Using hardcoded backbone ckpt: {use_ckpt_path}", flush=True)
    else:
        use_ckpt_path = args.model_path
        if is_master:
            print(f"[WARN] Hardcoded ckpt not found; falling back to provided model_path: {use_ckpt_path}", flush=True)

    try:
        ckpt = torch.load(use_ckpt_path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(use_ckpt_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    # 冻结主干
    for p in model.parameters(): p.requires_grad = False
    for p in model.norm.parameters(): p.requires_grad = True
    for p in model.performer.net.layers[-2].parameters(): p.requires_grad = True
    # 可选：再放开最后一层
    # for p in model.performer.net.layers[-1].parameters(): p.requires_grad = True

    # ---- 创建分类头 ----
    model.to_out = Identity(dropout=args.dropout, h_dim=128, out_dim=label_dict.shape[0], use_bayes=use_bayes_head)

    # **关键修复**：如果使用 Bayes 头，尽可能从 A 阶段确定性头初始化 μ（logvar 置小）
    if use_bayes_head:
        try:
            ported = try_port_ckpt_head_to_bayes(model, ckpt, is_master=is_master)
            if not ported and is_master:
                print("[WARN] Deterministic head weights not found in ckpt or porting failed; Bayes head stays random.", flush=True)
        except Exception as e:
            print(f"[WARN] Bayes-head init from A failed: {e}", flush=True)

    model = model.to(device)
    if ddp:
        model = DDP(
            model,
            device_ids=[LOCAL_RANK] if use_cuda else None,
            output_device=LOCAL_RANK if use_cuda else None,
            find_unused_parameters=False,
        )

    # 优化器：主干与头分组，头的 LR 更小
    head = get_model_head(model)
    base_params = [p for n, p in model.named_parameters() if 'to_out' not in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'to_out' in n and p.requires_grad]
    try:
        optimizer = Adam([
            {'params': base_params, 'lr': LEARNING_RATE},
            {'params': head_params, 'lr': LEARNING_RATE * head_lr_mult}
        ], weight_decay=args.weight_decay)
    except Exception:
        optimizer = Adam([
            {'params': base_params, 'lr': LEARNING_RATE},
            {'params': head_params, 'lr': LEARNING_RATE * 0.5}
        ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, first_cycle_steps=15, cycle_mult=2,
        max_lr=LEARNING_RATE, min_lr=1e-6, warmup_steps=5, gamma=0.9
    )

    # 类权重
    classes = np.arange(len(label_dict))
    cls_weight = compute_class_weight(class_weight='balanced', classes=classes, y=label_train.numpy())
    cls_weight_t = torch.tensor(cls_weight, dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=cls_weight_t).to(device)

    if ddp: dist.barrier()
    trigger_times = 0
    max_acc = 0.0

    if is_master:
        hp_summary = (
            f"HP SUMMARY | LR={LEARNING_RATE} | BATCH={BATCH_SIZE} | GRAD_ACC={GRADIENT_ACCUMULATION} | "
            f"BAYES={use_bayes_head} | KL_W={kl_weight} | KL_ANN={kl_anneal_steps} | MC_SAMPLES={bayes_mc_samples} | "
            f"PRE_WARM={pre_warm_epochs} | L1={l1_lambda} | DROPOUT={args.dropout} | WD={args.weight_decay} | "
            f"PRUNE_TARGET={prune_target}"
        )
        print(hp_summary, flush=True); logging.info(hp_summary)

    # ----------- 训练主循环 -----------
    for i in range(1, EPOCHS + 1):
        print(f"[INFO] Starting epoch {i} for fold {number_split}...", flush=True)

        if ddp and hasattr(train_loader, "sampler"):
            try:
                train_loader.sampler.set_epoch(i)
            except AttributeError:
                pass

        epoch_loss, epoch_acc = finetune_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            current_epoch=i, pre_warm_epochs=pre_warm_epochs, use_bayes_head_flag=use_bayes_head
        )

        if dist.is_available() and dist.is_initialized() and world_size > 1:
            epoch_loss = get_reduced(epoch_loss, LOCAL_RANK, 0, world_size)
            epoch_acc  = get_reduced(epoch_acc,  LOCAL_RANK, 0, world_size)

        if is_master:
            head = get_model_head(model)
            sp_elem = elem_sparsity(head, eps=1e-8)
            sp_struct, num_lin = 0.0, 0
            for _, lyr in head.named_children():
                if hasattr(lyr, "weight") or hasattr(lyr, "weight_mu"):
                    sp_struct += structured_sparsity_linear(lyr, dim=prune_dim, eps=1e-8)
                    num_lin += 1
            sp_struct = sp_struct / max(1, num_lin)

            log_line = (f'== Epoch: {i} | Training Loss: {epoch_loss:.6f} | '
                        f'Accuracy: {epoch_acc:6.4f}% | '
                        f'Sparsity(to_out elem)={sp_elem:.4f} | '
                        f'Sparsity(to_out struct~dim{prune_dim})={sp_struct:.4f} ==')
            print(log_line, flush=True); logging.info(log_line)

            try:
                # Do not write a partial line here; defer writing a single combined CSV row until after validation
                pass
            except Exception as e:
                logging.warning(f"Failed to write training metrics to {metrics_file}: {e}")

        if ddp: dist.barrier()
        scheduler.step()

        # 验证
        if i % VALIDATE_EVERY == 0:
            model.eval()
            if ddp: dist.barrier()

            # **关键修复**：预热期评估用 mc=1，之后再用设置的 MC 样本数
            mc_for_eval = 1 if (use_bayes_head and i <= pre_warm_epochs) else bayes_mc_samples

            running_loss = 0.0
            predictions, truths = [], []
            with torch.no_grad():
                for index, (data_v, labels_v) in enumerate(val_loader, start=1):
                    data_v, labels_v = data_v.to(device), labels_v.to(device)
                    logits = forward_eval(model, data_v, mc=mc_for_eval)
                    loss = loss_fn(logits, labels_v)
                    running_loss += loss.item()
                    final_prob = logits.softmax(dim=-1)
                    conf, final = final_prob.max(dim=-1)
                    if UNASSIGN_THRES > 0.0:
                        final[conf < UNASSIGN_THRES] = -1
                    predictions.append(final); truths.append(labels_v)

            del data_v, labels_v, logits, final_prob, final

            preds_tensor  = torch.cat(predictions, dim=0) if len(predictions) > 0 else torch.tensor([], device=device)
            truths_tensor = torch.cat(truths, dim=0) if len(truths) > 0 else torch.tensor([], device=device)
            total_size = len(val_loader.dataset)

            if dist.is_available() and dist.is_initialized() and world_size > 1:
                predictions = distributed_concat(preds_tensor,  total_size, world_size)
                truths      = distributed_concat(truths_tensor, total_size, world_size)
            else:
                predictions = preds_tensor[:total_size]
                truths      = truths_tensor[:total_size]

            no_drop = predictions != -1
            predictions = np.array((predictions[no_drop]).cpu())
            truths = np.array((truths[no_drop]).cpu())
            cur_acc = accuracy_score(truths, predictions) if truths.size > 0 else 0.0
            f1 = f1_score(truths, predictions, average='macro') if truths.size > 0 else 0.0

            val_loss = running_loss / max(1, len(val_loader))
            if dist.is_available() and dist.is_initialized() and world_size > 1:
                val_loss = get_reduced(val_loss, LOCAL_RANK, 0, world_size)

            if is_master:
                log_line = f'== Epoch: {i} | Validation Loss: {val_loss:.6f} | Validation Accuracy: {cur_acc*100:.2f}% | F1 Score: {f1:.6f} =='
                print(log_line, flush=True); logging.info(log_line)

                conf_matrix = confusion_matrix(truths, predictions)
                try:
                    class_report = classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4, zero_division=0)
                except Exception:
                    class_report = classification_report(truths, predictions, digits=4, zero_division=0)

                logging.info(f"Confusion Matrix:\n{conf_matrix}")
                logging.info(f"Classification Report:\n{class_report}")
                print(f"Confusion Matrix:\n{conf_matrix}", flush=True)
                print(f"Classification Report:\n{class_report}", flush=True)

                # Resource usage and timing
                gpu_usage = torch.cuda.memory_allocated(device) / 1e6 if use_cuda else 0.0
                memory_usage = torch.cuda.max_memory_allocated(device) / 1e6 if use_cuda else 0.0
                compute_time = time.time() - start_time

                # Global params & sparsity
                total_params = sum(p.numel() for p in model.parameters())
                nonzero_params = sum((p.abs() > 1e-8).sum().item() for p in model.parameters())
                global_sp = elem_sparsity(model, eps=1e-8)

                # Write a single combined CSV row for this epoch atomically:
                try:
                    # compute KL diagnostics for reporting
                    head = get_model_head(model)
                    if use_bayes_head:
                        try:
                            kl_term = float(head.kl().detach().cpu().item())
                        except Exception:
                            kl_term = 0.0
                        kl_scale = min(1.0, global_step / max(1, kl_anneal_steps)) * kl_weight
                    else:
                        kl_term = 0.0
                        kl_scale = 0.0
                    # include KL term and KL scale in CSV and logs
                    logging.info(f"KL term: {kl_term:.6f}, KL_scale: {kl_scale:.6f}")
                    print(f"KL term: {kl_term:.6f} | KL_scale: {kl_scale:.6f}", flush=True)
                    # Ensure column order keeps val_acc (col5) and f1 (col6); leave post-prune fields empty here
                    line = (
                        f"{i},{epoch_loss:.6f},{epoch_acc:.4f},{val_loss:.6f},{cur_acc:.6f},{f1:.6f},,"
                        f"{total_params},{nonzero_params},{sp_elem:.4f},{sp_struct:.4f},{global_sp:.4f},"
                        f"{kl_term:.6f},{kl_scale:.6f},{gpu_usage:.2f},{memory_usage:.2f},{compute_time:.2f}\n"
                    )
                    _atomic_append_line(metrics_file, line)
                except Exception as e:
                    logging.warning(f"Failed to write combined metrics to {metrics_file}: {e}")
                print(f"GPU Usage: {gpu_usage:.2f} MB | Memory Usage: {memory_usage:.2f} MB | Compute Time: {compute_time:.2f} s", flush=True)

            if cur_acc > max_acc:
                max_acc = cur_acc
                trigger_times = 0
                save_best_ckpt(i, model, optimizer, scheduler, val_loss, cur_model_name, ckpt_dir)
            else:
                trigger_times += 1
                if trigger_times > PATIENCE:
                    break
    # end epoch loop

    # ----------- 渐进式结构化剪枝 -----------
    loss_fn_ft = nn.CrossEntropyLoss(weight=cls_weight_t).to(device)
    prune_info = progressive_structured_prune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn_ft,
        device=device,
        world_size=world_size,
        val_sampler=val_sampler,
        target=prune_target,
        step=prune_step,
        max_ft_epochs=prune_ft_epochs,
        f1_drop_tol=prune_f1_tol,
        dim=prune_dim,
        layer_names=prune_layers,
        is_master=is_master,
        mc_eval=bayes_mc_samples
    )
    prune_best_f1 = prune_info.get('best_f1') if isinstance(prune_info, dict) else prune_info

    # ----------- 剪后立刻评估 & 稀疏率汇总 -----------
    if is_master:
        # try to get accurate per-block peak memory by resetting and syncing
        try:
            if use_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        tr_acc, tr_f1, _ = quick_eval(model, train_loader, device, world_size, val_sampler=train_sampler, mc=bayes_mc_samples, loss_fn=loss_fn)
        va_acc, va_f1, _ = quick_eval(model, val_loader,   device, world_size, val_sampler=val_sampler,   mc=bayes_mc_samples, loss_fn=loss_fn)

        # get measured peak after evaluations
        try:
            gpu_usage_peak = torch.cuda.max_memory_allocated(device) / 1e6 if use_cuda else 0.0
        except Exception:
            gpu_usage_peak = 0.0

        head = get_model_head(model)
        sp_elem = elem_sparsity(head, eps=1e-8)
        sp_struct, num_lin = 0.0, 0
        for _, lyr in head.named_children():
            if hasattr(lyr, "weight") or hasattr(lyr, "weight_mu"):
                sp_struct += structured_sparsity_linear(lyr, dim=prune_dim, eps=1e-8)
                num_lin += 1
        sp_struct = sp_struct / max(1, num_lin)

        print(f"[Post-Prune Summary] Train Acc={tr_acc:.4f}, Train F1={tr_f1:.4f} | Val Acc={va_acc:.4f}, Val F1={va_f1:.4f} | "
              f"Sparsity(elem)={sp_elem:.4f}, Sparsity(struct~dim{prune_dim})={sp_struct:.4f}")

        try:
            # Append a post-prune summary row filling the post_prune fields.
            last_epoch = i if 'i' in globals() else EPOCHS
            post_prune_ft_f1 = f"{va_f1:.6f}"
            prune_done_best_f1 = f"{prune_best_f1:.6f}" if prune_best_f1 is not None else ''

            total_params = sum(p.numel() for p in model.parameters())
            nonzero_params = sum((p.abs() > 1e-8).sum().item() for p in model.parameters())
            global_sp = elem_sparsity(model, eps=1e-8)
            # use peak GPU usage measured for this block
            gpu_usage = gpu_usage_peak
            memory_usage = torch.cuda.max_memory_reserved(device) / 1e6 if use_cuda else 0.0
            compute_time = time.time() - start_time

            # For post-prune row: many training/val fields are blank; include KL fields as blank as well
            line = (
                f"{last_epoch},,,,,,{post_prune_ft_f1},{prune_done_best_f1},{total_params},{nonzero_params},{sp_elem:.4f},{sp_struct:.4f},{global_sp:.4f},,"
                f"{gpu_usage:.2f},{memory_usage:.2f},{compute_time:.2f}\n"
            )
            _atomic_append_line(metrics_file, line)
        except Exception as e:
            logging.warning(f"Failed to append post-prune summary: {e}")

        # Additional concise summary to highlight Stage-B benefits
        try:
            pre_acc = prune_info.get('pre_acc') if isinstance(prune_info, dict) else None
            pre_f1 = prune_info.get('pre_f1') if isinstance(prune_info, dict) else None
            pre_pruned = prune_info.get('final_pruned') if isinstance(prune_info, dict) else None

            real_model = model.module if isinstance(model, DDP) else model
            # save temporary state to measure model file size (MB)
            import tempfile
            fd, tmpckpt = tempfile.mkstemp(suffix='.pth')
            os.close(fd)
            try:
                torch.save(real_model.state_dict(), tmpckpt)
                model_size_mb = os.path.getsize(tmpckpt) / 1e6
            except Exception:
                model_size_mb = 0.0
            finally:
                try: os.remove(tmpckpt)
                except Exception: pass

            print('\n=== Stage-B Summary (concise) ===')
            if pre_acc is not None:
                print(f"Pre-prune Val Acc/F1: {pre_acc:.4f} / {pre_f1:.4f}")
            print(f"Post-prune Val Acc/F1: {va_acc:.4f} / {va_f1:.4f}")
            if pre_pruned is not None:
                print(f"Total structured pruned (approx): {pre_pruned:.2f}")
            print(f"Params: total={total_params:,}, nonzero={nonzero_params:,}, global_sparsity={global_sp:.4f}")
            print(f"Head sparsity (to_out): elem={sp_elem:.4f}, struct_dim{prune_dim}={sp_struct:.4f}")
            print(f"Model file size (state_dict): {model_size_mb:.2f} MB")
            print(f"Measured GPU peak during eval: {gpu_usage:.2f} MB | reserved: {memory_usage:.2f} MB")
            # If Bayes head, report MC uncertainty approx: run small MC probe
            if use_bayes_head:
                try:
                    # one-batch MC probe on val loader first batch
                    probe_iter = iter(val_loader)
                    probe_x, probe_y = next(probe_iter)
                    probe_x = probe_x.to(device)
                    # reset peak and collect before probe
                    if use_cuda:
                        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
                    gc.collect(); torch.cuda.empty_cache()
                    logits_list = []
                    with torch.no_grad():
                        for _ in range(min(8, bayes_mc_samples)):
                            out = forward_eval(model, probe_x, mc=1)
                            logits_list.append(out.softmax(dim=-1).cpu().numpy())
                    import numpy as _np
                    prob_mean = _np.mean(_np.stack(logits_list, axis=0), axis=0)
                    prob_std = _np.std(_np.stack(logits_list, axis=0), axis=0)
                    avg_uncert = float(_np.mean(prob_std))
                    print(f"Bayes MC probe: avg predictive std (per-class) ~ {avg_uncert:.6f}")
                except Exception:
                    pass
            print('=== End Stage-B Summary ===\n')
        except Exception as e:
            logging.warning(f"Failed to print Stage-B concise summary: {e}")
# end folds

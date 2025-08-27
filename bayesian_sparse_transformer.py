#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from bayesian_sparse_transformer import SpatialBayesianTransformer
import scanpy as sc
import anndata as ad

class Tee(object):
    """同时输出到文件和屏幕"""
    def __init__(self, name, mode='w'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

class SCDataset(Dataset):
    """内存优化的单细胞数据集"""
    def __init__(self, data, label, top_genes=5000):
        self.data = data
        self.label = label
        self.top_genes = top_genes
        self.spatial_coords = np.random.rand(data.shape[0], 2)
        
    def __getitem__(self, index):
        row = self.data[index].toarray()[0] if hasattr(self.data[index], 'toarray') else self.data[index]
        # 动态选择高表达基因
        keep_genes = np.argsort(row)[-self.top_genes:]  
        sparse_gene = torch.zeros(self.data.shape[1])
        sparse_gene[keep_genes] = torch.FloatTensor(row[keep_genes])
        return (
            sparse_gene,
            torch.FloatTensor(self.spatial_coords[index]),
            torch.LongTensor([self.label[index]])
        )

    def __len__(self):
        return self.data.shape[0]

def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_num", type=int, default=16906, help='Total number of genes')
    parser.add_argument("--batch_size", type=int, default=2, help='Physical batch size')
    parser.add_argument("--grad_acc", type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument("--spatial_dim", type=int, default=128, help='Spatial encoding dimension')
    parser.add_argument("--kl_weight", type=float, default=0.1, help='Weight for KL divergence')
    parser.add_argument("--model_name", default='optimized_model', help='Model save name')
    parser.add_argument("--data_path", default='./data.h5ad', help='Path to h5ad file')
    parser.add_argument("--top_genes", type=int, default=5000, help='Number of top genes to keep')
    args = parser.parse_args()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 日志设置
    log_file = f"{args.model_name}.log"
    tee = Tee(log_file)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    # 数据加载
    adata = sc.read_h5ad(args.data_path)
    label_dict, label = np.unique(adata.obs['cell_type'], return_inverse=True)
    train_dataset = SCDataset(adata.X, label, args.top_genes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 模型初始化
    model = SpatialBayesianTransformer(
        gene_dim=args.gene_num,
        spatial_dim=args.spatial_dim,
        num_classes=len(label_dict),
        num_tokens=args.bin_num + 2,
        max_seq_len=args.gene_num + 1
    ).to(device)

    # 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = GradScaler()  # 混合精度

    # 训练循环
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        for i, (genes, coords, labels) in enumerate(train_loader):
            # 混合精度前向
            with autocast():
                logits = model(genes.to(device), coords.to(device))
                logits = logits[:, 0, :]
                loss = (loss_fn(logits, labels.to(device)) / args.grad_acc
            
            # 梯度累积
            scaler.scale(loss).backward()
            
            if (i+1) % args.grad_acc == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        # 验证和保存逻辑
        if epoch % 5 == 0:
            validate(model, val_loader, device)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f"{args.model_name}_epoch{epoch}.pth")

if __name__ == "__main__":
    main()
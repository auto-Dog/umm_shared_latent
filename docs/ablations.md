# Ablation protocol

所有实验固定：数据样本 ID/seed、图像分辨率、采样步数、有效 batch size、优化器、学习率计划与评测 prompt。不要用训练 wall time 作为唯一算力口径；同时记录 GPU-hours、峰值显存、trainable parameters 和推理延迟。

## 第一轮：baseline 可训练性

| ID | Query | Connector | Loss | 配置 |
|---|---:|---|---|---|
| B0 | 256 | 6-layer Light Transformer | Flow | `baseline_pt.yaml` → `legacy/baseline_dm.yaml` |
| A1 | 256 | Linear | Flow | `legacy/ablations/pt_linear_connector.yaml` → `legacy/ablations/dm_linear_connector.yaml` |
| A2 | 64 | 6-layer Light Transformer | Flow | `legacy/ablations/pt_query64.yaml` → `legacy/ablations/dm_query64.yaml` |
| L1 | 256 | 6-layer Light Transformer | Flow + latent alignment | `legacy/ablations/pt_alignment_loss.yaml` → `legacy/ablations/dm_alignment_loss.yaml` |
| L2 | 256 | 6-layer Light Transformer | Flow + query diversity | `legacy/ablations/pt_diversity_loss.yaml` → `legacy/ablations/dm_diversity_loss.yaml` |

第一轮只判断架构/损失是否有稳定增益。若 L1/L2 无显著改善，不进入大数据或 RL 阶段。

## 评测顺序

1. 训练侧：loss 曲线、梯度范数、Query 两两 cosine、峰值显存、GPU-hours。
2. T2I 主指标：GenEval；补充 DPG-Bench、WISE、COCO/MJHQ FID。
3. 编辑/条件生成：保持、指令遵循和 artifact 类指标。
4. 理解保真：MME-P、MMBench、MMMU、MM-Vet；冻结 Qwen 时应验证能力没有回退。
5. 只有 baseline/消融完成后，才增加 SFT、文本可验证奖励、Flow-GRPO、图文比例与 anti-hacking reward。

至少运行 3 个 seed；报告均值和标准差。文档当前没有任何本地训练结果，因此这里不填写提升数值。

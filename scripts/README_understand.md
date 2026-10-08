# 多模态理解：PreFill 阶段注入 learnable query 的实验

实验日期：2026-10-08。检查点：`t2i_cc12m_pt_1007_16667`（cc12m 预训练）与 `t2i_blip3o_sft_1007_9000`（blip3o SFT）。GPU 1，conda `uniquery`，脚本 `scripts/run_understand.py`。

## 机制

UMM(MetaQuery) 结构：冻结 InternVL3-1B LLM + 256 个可学习 query token（`<begin_of_img><img0..255><end_of_img>`，即 `model.query_suffix`）+ 6 层 connector。图像经动态 448 预处理后由 ViT `extract_feature` 得到 `(num_patches, 256, C)` 特征，注入到 user 轮 `<IMG_CONTEXT>` 位置。随后测试两种**在 PreFill 阶段直接注入 learnable query** 的位置：

1. **assistant 轮末尾**（T2I 训练时 query 所在位置）：`...user...<|im_start|>assistant\n` + `query_suffix`
2. **user 轮末尾**（`<|im_end|>` 之前）：`<image>+问题` + `query_suffix` + `<|im_end|>`

causal 注意力下两种位置 query 都能 attend 到图片+问题（query 在它们之后），区别是 assistant 轮末尾之后模型需要直接续写文本，而 user 轮末尾之后 assistant 轮是正常的文本续写先验。对比基线：不注入 query 的普通 VQA 格式。全部 greedy 解码，`model.ivl3.generate()`（官方包装，内部完成 ViT 特征注入）。

## 结果（5 个问答对 × 3 变体）

测试图片：`/home/mingjun/OpenUni/OpenUni_MME/models/InternVL3-1B/examples/`（image1 = 红熊猫坐木板，image2 = 人喂大熊猫竹子）。

| 注入位置 | cc12m PT（`results_cc12m_*.json`） | SFT（`results_sft_*.json`） |
|---|---|---|
| 基线（不注入） | 5/5 答对，简短准确 | 5/5 答对，简短准确 |
| **assistant 轮末尾** | **0/5**：全部退化为重复乱码（"local本地 local本地…"、"強強強…"、"exexex…"），第 2 问空串 | **0/5**：同样退化为重复乱码（"uousuous…"、"conceptusconceptus…"、"lying…"） |
| **user 轮末尾（`<|im_end|>` 前）** | **4/5 答对**，回答更长更完整（"red panda"、"wooden platform"、"reddish-brown coat"）；第 5 问开头答对但尾部掉入 "Loc Loc…" 重复 | **5/5 答对且稳定**，回答完整流畅，无重复崩溃 |

## 结论

1. **直接回答"在 user `<|im_end|>` 之前加 query 会怎样"**：这是**正确的注入位置**。query 在图片+问题之后、`<|im_end|>` 之前，PreFill 阶段就能 attend 到图片 token 与问题，随后 `<|im_start|>assistant\n` 是一个干净的文本续写起点。cc12m PT 与 SFT 两个 checkpoint 在此位置均能正常作答。
2. **assistant 轮末尾注入不可行**：LLM 从未被训练在 query token 序列之后续写文本（T2I 方向 query 之后是扩散解码而非文本生成），注入后贪心解码直接塌缩成重复 token——两个 checkpoint 都如此，说明这不是权重差异而是位置本身 OOD。
3. query 注入 vs 基线的语义增益：5 个样本上两者都答对，注入版回答信息更全（如 Q1 给出完整描述、Q2 "wooden platform" 比基线更确定）。是否真的"受益于 image token"需要更大样本/评测集（如 MME 子集 + 有/无 query 对比）来量化。
4. SFT 相比 cc12m PT：user 轮末尾注入时更稳定（cc12m 第 5 问的尾部重复在 SFT 上消失），说明 SFT 进一步收敛了 query 在理解方向的行为。

## 复现

```bash
cd /home/mingjun/umm_uniquery
UNDERSTAND_POS=user   CFG10_GPU=1 python scripts/run_understand.py /home/mingjun/models/t2i_cc12m_pt_1007_16667   # → results_cc12m_user.json
UNDERSTAND_POS=user   CFG10_GPU=1 python scripts/run_understand.py /home/mingjun/models/t2i_blip3o_sft_1007_9000   # → results_sft_user.json
UNDERSTAND_POS=assistant CFG10_GPU=1 python scripts/run_understand.py <ckpt>                                       # 对照实验
```

# Baseline architecture

## 数据流

```text
text / reference images
          │
          ▼
 frozen Qwen2.5-VL-3B（MetaQuery BOI + 256 `<img_i>` + EOI）
          │
          ▼
 取 256 个 Query hidden states
 + 6 × (bidirectional self-attn → SwiGLU)
          │
          ▼
 projection to Sana caption channels
          │
          ▼
 frozen Sana-0.6B + frozen DC-AE ── flow matching loss
```

Light Connector 使用 896 hidden size、4096 intermediate size、14 heads 和 6 layers。按 Qwen hidden size 2048、Sana caption channels 2304 估算，Query + Connector 约 90M 参数，处于“约 0.1B”的预算内。与原始 MetaQuery 相同，先扩展 Qwen embedding，再把 `<begin_of_img>`、256 个 `<img_i>` 和 `<end_of_img>` 追加到因果序列末尾；embedding 梯度钩子将原词表行的梯度清零，仅更新新增行。通过 Qwen2.5-VL 顶层官方 forward 后，只提取 BOI/EOI 之间的 Query hidden states，再送入双向 Connector。Qwen 其余参数与 Sana 均冻结，但反向图会穿过两者；VAE 编码保持 `no_grad`。

Qwen2.5-VL 的 `max_pixels` 在配置校验和 Processor 初始化时均硬限制为 `1,000,000`。Processor 会按视觉 patch/grid 规则缩放，因此实际进入模型的像素数只会小于或等于该值。

带图 forward 与原始 MetaQuery 一致，直接调用 `Qwen2_5_VLForConditionalGeneration(...)` 顶层接口并传入 `pixel_values` 与 `image_grid_thw`；将 `lm_head` 设为 `Identity`，从 `.logits` 取得最后一层 hidden states，避免生成词表 logits。进入 forward 前仍保留 patch、grid、image token 和 1,000,000 pixels 上限校验。训练关闭 Accelerate `dispatch_batches`，避免跨 rank 拼接变长视觉张量。

与参考 MetaQuery 的差别是：参考实现使用 24 层双向 Qwen2 Encoder，并可微调 Sana；本 baseline 为满足“可训练部分约 0.1B”采用 6 层双向 self-attention 连接器，且冻结 Sana。该差异被配置显式记录，避免把参数预算不一致的结果混为同一 baseline。

## Loss

默认目标只有：

```text
L = L_flow
```

两个实验损失均默认关闭：

- `latent_alignment_weight`：由 query 均值预测目标 VAE latent 的 channel mean/std；
- `query_diversity_weight`：惩罚归一化 query Gram matrix 的非对角项，观察 query collapse。

两者不会污染 baseline，可通过独立 YAML 开启。

## Checkpoint contract

adapter checkpoint 只包含以下 key 前缀：

```text
metaquery_embeddings
connector.*
alignment_head.*
```

`metaquery_embeddings` 是 Qwen embedding 中 BOI、EOI 和 256 个 `<img_i>` 的连续新增行；adapter-only checkpoint 不复制完整 Qwen embedding。

基础模型 ID、架构和损失配置保存在 `adapter_config.json`。DM、后续 SFT/RL 必须从同一基础模型 ID 加载 adapter；如果改变 Query 数或 Connector 宽度，应作为新实验而非 strict resume。

## Extension contract

- 数据源：在 `DATA_SOURCES` 注册 raw-row → unified-example 转换器；
- 训练阶段：在 `STAGES` 注册 builder；
- SFT：在 `SFT_FORMATTERS` 注册 reasoning/answer label formatter；
- RL：在 `REWARD_BUILDERS` 注册文本、图像或图文平衡奖励；
- 模型：复用公开方法 `encode_queries()` 与 `compute_flow_loss()`，避免让 RL 代码依赖内部 forward 细节。

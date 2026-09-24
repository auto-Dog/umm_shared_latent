# InternVL3-1B backbone + 本地数据提速方案

## 目标
当前单卡 GPU1 的 Qwen2.5-VL-3B 版跑通但太慢（~40s/步，数据管线是瓶颈）。参考
`/home/mingjun/OpenUni/work_dirs/openuni_b_stream_pt/..._stream_pt.py`，做两件事：
1. **新增 InternVL3-1B backbone 类**（自包含拷贝进本项目，用户已拍板），复用本项目
   的 MetaQuery special-token 机制（**不是** OpenUni 的独立 `meta_queries` 参数）。
2. **数据管线提速**：本地 cc12m webdataset tar + batch 8 + 关 gradient checkpointing
   + flash attention。

## 架构对照（为什么这样移植）

| 组件 | 本项目 Qwen 版 | OpenUni InternVL3 版 | 本方案 InternVL3 版 |
|---|---|---|---|
| LLM | Qwen2.5-VL-3B（冻结） | InternVL3-1B Llama（冻结） | InternVL3-1B（冻结） |
| MetaQuery | **special token** BOI/EOI + N query 词表行 | 独立 `meta_queries` 参数拼 embeds | **special token**（同 Qwen 版，用户要求） |
| query 定位 | BOI/EOI 区间取 hidden | `last_hidden_state[:, -N:]` | BOI/EOI 区间取 hidden |
| 图像 patch | Qwen `pixel_values+image_grid_thw` 外部注入 | `<IMG_CONTEXT>` forward 内部替换 vit_embeds | `<IMG_CONTEXT>` forward 内部替换 |
| connector | `light_transformer`（context→hidden→output 双投影） | OpenUni `ConnectorEncoder` + projector | 复用本项目 `build_connector` |
| checkpoint | `adapter_model.safetensors`（connector + metaquery 行） | — | 同 Qwen 版格式（完全兼容） |

## 改动清单

### 1. 拷贝 InternVL3 模型代码（自包含）
新建 `src/umm_uniquery/modeling/internvl3/`，从 `/home/mingjun/OpenUni/src/models/internvl3/`
拷贝 5 个文件（MIT 许可，本项目 `NOTICE.md` 需加出处）：
- `modeling_internvl_chat.py`
- `modeling_intern_vit.py`
- `configuration_internvl_chat.py`
- `configuration_intern_vit.py`
- `conversation.py`
（`debug.py` 不需要。`flash_attn` 可选依赖，未装则 fallback eager/sdpa。）

改造 `modeling_internvl_chat.py`：
- 前向加 `image_flags` 默认支持（OpenUni 版在 `forward` 里 `squeeze` 需要；本项目 collator
  不产 image_flags，需改为：图像存在与否由 `pixel_values` 是否为 None 决定，或保留参数并默认全 1）。
- **关键改动**：语言模型前向后取 hidden states 的接口。OpenUni 版通过
  `llm.model(**inputs, return_dict=True).last_hidden_state` 取（绕开 lm_head）。
  本项目 Qwen 版是 `lm_head = Identity()` 后走 `output.logits`。InternVL3 的
  `language_model` 是原生 LlamaForCausalLM，**直接在类外调用 `language_model.model(...)`**
  取 `last_hidden_state`，不动 lm_head（避免改预训练权重结构）。

### 2. 新增模型类 `UniQueryInternVL3Model`（`src/umm_uniquery/modeling/model.py` 同文件）
对齐 Qwen 版 `UniQueryModel` 的对外接口（`train.py`/`trainer.py` 零改动）：
- `__init__`：加载 InternVL3（`InternVLChatModel.from_pretrained(ivl3_id, use_flash_attn=...)`），
  tokenizer（本地路径 `IVL3_ID`），Sana/VAE/scheduler（复用现有逻辑），
  `build_connector`，special-token MetaQuery 词表构造（**复制 Qwen 版 52-108 行逻辑**，
  resize `language_model.get_input_embeddings()` + tokenizer 对齐 + BOI/EOI/query token 连续性断言），
  `requires_grad_(False)` + 冻结原 embedding 行的 hook。
- `encode_context`：InternVL3 版——文本 `input_ids` 走 `language_model.model()`
  拿到 `last_hidden_state`，BOI/EOI 区间切片取 `[B, N, hidden]`。图像 patch
  **不进 text2image 的 query 路径**（OpenUni 的 t2i 路径也不喂图像；图像只在
  image2image 路径用，本项目当前 t2i 阶段不需要）。故 `encode_context` 只需
  `input_ids` + `attention_mask`。
- `compute_flow_loss` / `forward` / `pixels_to_latents` / `generate_t2i`：
  直接复用 Qwen 版实现（Sana 侧完全同构），生成时用
  `metaquery_embeddings` 切片拼 BOI/EOI prompt（同 Qwen 版 `render_prompt`，仅
  prompt template 差异：InternVL 用 `<|im_start|>` chat template）。
- `adapter_state_dict` / `save_adapter` / `load_adapter` / `trainable_parameter_count`：
  完全复用 Qwen 版（connector. + alignment_head. + metaquery_embeddings 行），
  格式兼容 → 两 backbone 的 checkpoint 可互读（shape 若不同则需 init_checkpoint 匹配）。

### 3. Processor/collator 适配（`src/umm_uniquery/data/collator.py`）
InternVL3 的 collator：
- `processor` 换成 InternVL3 的 tokenizer（无 Qwen processor；图像预处理用
  OpenUni 的 IMAGENET normalize + resize 448，`_center_crop_resize` 已产出
  [-1,1] 张量，可复用，再加 normalize/interpolate 到 448）。
- `_prompt`：InternVL3 chat template（`<|im_start|>user\n...<|im_end|>assistant\n`）+
  `query_suffix`（BOI + query tokens + EOI）。
- `__call__` 输出：`input_ids` + `attention_mask`（tokenizer 直接编，无需
  Qwen 的 images/`image_grid_thw`）+ `target_pixels`。
- **collator 与模型类解耦**：新增 `collator_type` 配置项（`"qwen"` / `"internvl3"`），
  `stages.py` 按 `model.backbone` 或 `collator_type` 分派。

### 4. 数据管线提速
- 新增 webdataset 源支持：`streaming.py` 注册 `cc12m_wds` source，指向
  `/data/mingjun/cc12m_local/*.tar`（`datasets` 的 `load_dataset("webdataset")`，
  `streaming=True`，或直接用 `webdataset` 库流式读 tar——本机已下载 4 个 tar，2.1GB）。
- 新配置 `configs/local_pt_ivl3.yaml`：
  ```yaml
  extends: local_pt.yaml
  model:
    backbone: internvl3
    ivl3_id: /home/mingjun/OpenUni/OpenUni_MME/models/InternVL3-1B
    attention_backend: flash_attention_2   # fallback sdpa if flash_attn absent
    gradient_checkpointing: false           # Sana 冻结时不需要
    trainable_parameter_guard_m: null
  data:
    sources:
      - kind: cc12m_wds
        path: /data/mingjun/cc12m_local/*.tar
        sample_count: 120000
        seed_offset: 0
    shuffle_buffer: 500
  training:
    per_device_train_batch_size: 8         # 本地数据，直接大 batch
    gradient_accumulation_steps: 1
    learning_rate: 5e-5
    warmup_ratio: 0.05
    max_steps: 1000
    output_dir: outputs/pt_ivl3
    run_name: uniquery-pt-ivl3
  ```

### 5. `train.py` / `trainer.py` / `config.py` 兼容
- `modeling/__init__.py` 导出 `UniQueryInternVL3Model`。
- `train.py` 按 `config["model"].get("backbone")` 选择类（默认 qwen）。
- `config.py` `validate_config`：InternVL3 分支跳过 Qwen 专属的 `max_pixels/min_pixels`
  校验，加 `backbone: internvl3` 时要求 `ivl3_id`。

## 验证步骤
1. 冒烟：`CONFIG=configs/local_pt_ivl3.yaml` + 小 sample_count（如 32），
   单卡 GPU1 跑 20 步，确认 loss 正常下降、trainable_parameter_count 与预期一致
   （connector ~0.1B 量级 + 258 行 embedding）。
2. 生成测试：`generate_t2i` 出一张图，肉眼确认非纯噪声。
3. 步速对比：本地 tar + batch 8 应远快于当前 ~40s/步；记录第一步 loss 时间。
4. 与 Qwen 版 checkpoint 交叉验证 `load_adapter` 形状检查逻辑。

## 风险与备注
- InternVL3 是 Llama 结构，`<IMG_CONTEXT>` 图像 patch 只有 image2image 才用；
  当前 t2i 阶段 collator 不产图像，`forward` 里 `pixel_values=None` 时 InternVL3
  的 `extract_feature` 不能调用 → **必须** 在 InternVL3 类前向里走
  `language_model.model(inputs_embeds=...)` 纯文本路径（OpenUni 的
  `InternVLChatModel.forward` 无条件调 `extract_feature(pixel_values)`，不能直接用，
  要绕过它）。这正好配合第 1 点的改造。
- `flash_attn` 未装（前面 pip 确认过 hf_transfer 时装的是别的包）：`flash_attention_2`
  fallback 到 sdpa，config 里用 `attention_backend` 配置。
- 本地只有 4 个 cc12m tar（2.1GB），sample_count 120k 够冒烟；全量 60k 数据集的
  tar 后续可续下。
- `config.py` 的 Qwen `max_pixels≤1M` 校验对 InternVL3 不适用，需按 backbone 分叉。

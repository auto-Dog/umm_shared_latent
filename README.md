# UMM UniQuery

这是一个面向最小数据对齐实验的 MetaQuery 风格 baseline。当前范围严格限定为：

- `Qwen/Qwen2.5-VL-3B-Instruct` 冻结骨干；
- Qwen2.5-VL 视觉输入硬限制为最多 1,000,000 pixels；
- 256 个 Learnable Query；
- 6 层 Light Connector（hidden size 896、14 heads、FFN 4096）；
- `Efficient-Large-Model/Sana_600M_512px_diffusers` 冻结生成骨干；
- PT 使用 CC12M 与 OmniEdit 的 1.2M 流式混合；
- DM 使用 MetaQuery-Instruct-2.4M 的 0.2M 流式抽样；
- 默认只有 Query、Connector 可训练，约 0.1B 参数；
- baseline 仅使用 Flow Matching，embedding/diversity 损失默认关闭。

模型核心采用 MetaQuery 最小差异实现：保留其“扩展 Qwen embedding + 冻结原词表行梯度”的 Learnable Query、`<begin_of_img>/<img_i>/<end_of_img>` token、Qwen2.5-VL 顶层 forward、BOI/EOI Query hidden-state 提取和 prompt 格式。差异仅保留在 6 层轻量 Connector、流式数据与抽样、损失扩展、训练/故障恢复设施，以及任务指定的 Sana-0.6B。

这里的 Qwen 选择严格沿用 MetaQuery 配置中的 `Qwen/Qwen2.5-VL-3B-Instruct`，因为它直接支持图文输入；模型将 `lm_head` 替换为 `Identity`，并从顶层 forward 的 `.logits` 读取最后一层 hidden states。

Qwen2.5-VL 的视觉 forward 对 Transformers 内部接口敏感，因此依赖固定为参考 MetaQuery 使用的 `transformers==4.49.0`。远端环境不要在不做兼容验证的情况下升级该版本。

## 项目结构

```text
configs/                 PT、DM、DeepSpeed 与消融配置
src/umm_uniquery/data/   精确配额的流式数据集与统一 collator
src/umm_uniquery/modeling/
                         Query、Light Connector、Sana Flow loss
src/umm_uniquery/training/
                         adapter-only Trainer 与 SFT/RL 协议
docs/                    架构和实验口径
```

## 数据口径

`baseline_pt.yaml` 将“CC12M + OmniEdit 抽样共 1.2M”明确化为 600K + 600K。这是当前唯一需要按实验资源调整的假设；修改两个 `sample_count` 即可，Loader 不会物化完整数据集。

流式 shuffle buffer 默认设为 500。远程流会先填满 buffer 才产出第一条数据；原来的 10,000 会在首个 batch 前下载约一万条图像，在受限网络或 HF mirror 上表现为长时间卡住。

流式 Loader 具有以下性质：

- Hugging Face `streaming=True`；
- 固定 seed 和 shuffle buffer；
- 按剩余配额随机混合，最终精确满足每个 source 的样本数；
- 坏图自动跳过并从同一流补齐；
- 不执行 `select()`、不下载完整 Arrow 数据；
- 关闭 Accelerate 的集中 batch 拼接，允许不同 rank 使用不同长度的文本与视觉 patch；
- 续训时数据流会自行快进（见下节），入口会据此自动设置 `ignore_data_skip=true`。

为保证 Accelerate 的进程分发与可恢复顺序，baseline 固定 `dataloader_num_workers: 0`。各 rank 会按确定性全局流选取自己的 batch；这会增加远端流读取量，但可以避免 T2I 无图样本与 Edit 有图样本在集中分发时发生变长张量拼接错误。I/O 扩展应优先增加远端 WebDataset shard 和节点缓存，不应直接增加 PyTorch worker。

### 续训数据快进

续训时不需要从第 0 条数据重放。`ExactStreamingMixture` 的 source 选择只由 seed 和各
source 的 `sample_count` 决定，与图像内容无关，因此可以在纯 Python 中 replay 这段
RNG，得到每个 source 已消费的条数，然后：

1. 按 shard 粒度丢弃已经消费完的 shard（不重新下载前缀，WebDataset/parquet 直接
   跳过大文件）；
2. 只在尚未消费完的那个边界 shard 上做行级 `.skip()`；
3. 用 replay 后的 RNG 状态继续按配额混合，保证 source 选择序列与连续训练完全一致。

恢复位置因此是 shard 粒度的“大致”位置：边界 shard 内最多会有 `shard_samples` 量级
的样本与完全连续训练不同，换来的是近乎零的续训启动成本。

入口 `train.py` 会从 `resume_from_checkpoint/trainer_state.json` 的 `global_step`
反推已消费条数 `(global_step - stage_start_step) × per_device_batch × grad_accum ×
world_size`，并在偏移大于 0 时自动把 `training.ignore_data_skip` 置为 `true`——
数据流已经快进过，若再让 Trainer 逐条 skip 会跳两次，在数据中留下空洞。

可调配置：

| 键 | 位置 | 作用 |
|---|---|---|
| `data_start_offset` | `training` | 直接指定已消费条数，覆盖从 checkpoint 反推的结果（例如换了 world size、或 `save_steps` 与 optimizer step 不线性对应时） |
| `stage_start_step` | `training` | 该 stage 的训练起始 step，用于把 `global_step` 换算为该 stage 已消费的数据量（多阶段接力时必填） |
| `shard_samples` | 单个 source | 每个 shard 的行数，用于决定能整块丢弃几个 shard；缺失则退化为行级 skip |
| `shard_start` | 单个 source | 手动从第 N 个 shard 开始读（忽略反推的偏移）；需要自行把该 source 的 `sample_count` 相应下调，否则会因为配额大于剩余数据而报错 |

## 预训练模型

训练与推理依赖的预训练权重（均为冻结骨干），按配置字段、来源仓库与用途列出，便于在新机器上复用下载：

| 配置字段 | 用途 | Hugging Face 仓库 |
|---|---|---|
| `mllm_id` | Qwen 版 backbone（Qwen2.5-VL 冻结 LLM + ViT） | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `ivl3_id` | InternVL3 版 backbone（InternVL3-1B 冻结，`configs/local_pt*.yaml` 使用） | `OpenGVLab/InternVL3-1B` |
| `sana_id` | Sana 0.6B 生成骨干（diffusers 完整 pipeline） | `Efficient-Large-Model/Sana_600M_512px_diffusers` |
| `vae_id` | Sana DC-AE VAE（32 latent channels，Flow Matching 重建损失用） | `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers` |

这份对应关系同时固化在 config 中：每个入口配置的顶层 `model_dependency` 段按字段记录了模型名称与官方仓库地址（`baseline_pt.yaml` 记 Qwen/Sana/VAE，`local_pt.yaml` 追加 InternVL3-1B，随 `extends` 递归合并）。`validate_config` 会校验配置引用的每个 `model.*_id` 字段都有对应条目。换机器时 `model.*_id` 可填本地下载目录，`model_dependency` 里的官方地址保持不变、以它为准重新下载。

在新机器上复用下载（建议走 `hf-mirror`）：

```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download OpenGVLab/InternVL3-1B \
  --local-dir /path/to/models/internvl3-1b
huggingface-cli download Efficient-Large-Model/Sana_600M_512px_diffusers \
  --local-dir /path/to/models/sana-600m-512px
huggingface-cli download mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers \
  --local-dir /path/to/models/sana-vae
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /path/to/models/qwen25-vl-3b-instruct
```

要点：

- InternVL3-1B 是 `internvl_chat` 结构（LLM 为 Qwen2.5 系、视觉为 InternViT），仓库自带自定义 modeling 源文件（`conversation.py`、`modeling_intern_vit.py` 等）。本仓库已将其内置到 `src/umm_uniquery/modeling/internvl3/`，配置只需 `ivl3_id` 指向权重与 tokenizer 目录。
- `sana-600m-512px` 的 `model_index.json` 为完整 SanaPipeline：`transformer=SanaTransformer2DModel`、`text_encoder=Gemma2Model`、`tokenizer=GemmaTokenizerFast`、`vae=AutoencoderDC`；`sana-vae` 即独立的 DC-AE（`AutoencoderDC`，latent_channels 32），代码中缺失时会回退 `AutoencoderKL`。
- Qwen 版视觉 forward 对 Transformers 接口敏感，`transformers` 固定 `4.49.0`；远端环境不要未经兼容验证升级该版本。
- `configs/local_pt*.yaml` 直接引用运行机上的本地模型目录（如 `/root/autodl-tmp/models/...`）；换机器时把上面下载到的本地目录填回 `ivl3_id` / `sana_id` / `vae_id` 即可，其余配置无需改动。

## 远端训练

以下命令仅供 GPU 训练节点使用，本项目未在当前本地机器执行：

```bash
cd umm_uniquery
python -m pip install -e '.[tracking]'

python -m umm_uniquery.resilient_launch --nproc-per-node=8 \
  --config configs/baseline_pt.yaml \
  --set training.output_dir=/mnt/checkpoints/uniquery_pt
```

DM 阶段从 PT adapter 初始化：

```bash
python -m umm_uniquery.resilient_launch --nproc-per-node=8 \
  --config configs/legacy/baseline_dm.yaml \
  --set model.init_checkpoint=/mnt/checkpoints/uniquery_pt \
  --set training.output_dir=/mnt/checkpoints/uniquery_dm
```

`max_steps` 默认由样本总量、world size、micro batch 和梯度累积自动计算。每个 checkpoint 只保存 Connector、Query 和可选 alignment head，不复制冻结的约 4B 基础权重。

### 失败自动恢复

推荐入口 `umm_uniquery.resilient_launch` 使用 TorchElastic 管理 worker group。任一 rank 出现 CUDA OOM、NCCL 异常、进程崩溃或被系统杀死时：

1. 终止当前所有 rank，释放 CUDA context；
2. 最多自动重启 `training.failure_recovery.max_restarts` 次；
3. 新进程扫描 `output_dir/checkpoint-*`；
4. 忽略保存中途损坏、缺文件或没有完成标记的 checkpoint；
5. 从最近完整 checkpoint 恢复 adapter、optimizer、scheduler、global step、RNG，并把确定性流式数据快进到已消费位置（按 shard 丢弃已消费 shard + 边界行级 skip），而不是逐条重放已消费前缀。

默认最多重启 5 次、每 5 秒检查一次。若 OOM 持续复现，超过上限后任务会失败退出，避免无限重启。一次故障最多损失最近 `save_steps` 以内的进度；需要更小恢复窗口时降低 `save_steps`，代价是更多 checkpoint I/O。

`auto_resume: true` 也意味着使用相同 `output_dir` 重新提交任务时会自动续训。需要全新实验时应使用新的空输出目录。若由外部调度器确保目录为空，也可显式设置：

```bash
--set training.failure_recovery.auto_resume=false
```

即使初次启动关闭了 `auto_resume`，TorchElastic 重启计数大于 0 时仍会恢复本次任务刚保存的 checkpoint，保证故障恢复语义不被关闭。

直接使用原始 `torchrun` 仍可自动发现 checkpoint，但不会自动重启失败进程；因此长训练建议始终使用 resilient launcher。

如果训练机通过 `HF_ENDPOINT=https://hf-mirror.com` 访问 Hub，入口会自动把 datasets-server 文件 URL和分页 `Link` 中残留的 `huggingface.co` 地址改写到该 mirror；默认 Hub 环境下这段兼容逻辑不会生效。`configs/local_pt.yaml` 和 `configs/local_pt_smoke.yaml` 保留了运行机本地权重路径及 600 条 smoke 配置，便于复现跑通检查。

远端生成 GenEval/DPG 等评测所需图片目录：

```bash
python -m umm_uniquery.sample \
  --config configs/legacy/baseline_dm.yaml \
  --checkpoint /mnt/checkpoints/uniquery_dm \
  --prompts /mnt/eval/geneval_prompts.jsonl \
  --output-dir /mnt/eval/uniquery_geneval
```

prompt 文件支持一行一个纯文本，或 `{"id": "...", "prompt": "..."}` JSONL。该入口当前只覆盖 T2I；编辑采样将在 baseline 验证后接入。

## 当前边界

PT/DM baseline 已实现；SFT 与 RL 只预留扩展接口，尚未伪造训练逻辑。后续分别通过 `SFT_FORMATTERS`、`REWARD_BUILDERS` 和 `STAGES` 注册，模型侧复用 `encode_queries()` 与 `compute_flow_loss()`。建议完成 baseline 与消融后再接入文本 SFT、Flow-GRPO 和图文平衡奖励。

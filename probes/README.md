# umm_uniquery / probes

单点测试（single-point probe）文件夹：对少量固定 prompt（当前基石为
`umbrella_baseball`，日后可追加其他 prompt），用多个模型各生成固定 seed 的
图片，便于横向对比不同模型 / prompt 增强 / 架构在同一 prompt 上的表现。

## 目录结构

```
probes/
├── prompts/             prompt 注册表（一行一个 JSON：id / prompt / note）
├── scripts/             每个模型一个生成脚本 + common.py + run_all.sh
├── outputs/<model>/<prompt_id>/
│   ├── <seed>.png       每 seed 一张 512x512 图
│   └── meta.jsonl       每样本一行（prompt、seed、参数、耗时）
└── logs/                各脚本 stdout
```

> **生成结果位置**：测试产物（`outputs/`、`logs/`、seed42 对比图）已移出本仓库，
> 统一放在 `/home/mingjun/probes/`。scripts / prompts / 本 README 仍在仓库内
> `probes/` 目录；重跑 `run_all.sh` 会就地重新生成 `probes/outputs/` 与
> `probes/logs/`。

## 当前模型

| 输出目录 | 模型 | 说明 |
|---|---|---|
| `outputs/sana512` | Sana_600M_512px（vanilla diffusers pipeline，Gemma2 text encoder） | 裸 Sana T2I baseline |
| `outputs/sana512_ivl3enh` | 同上 + InternVL3-1B prompt 增强 | prompt 先经 InternVL3-1B 改写（3 抽，语义校验失败回退原 prompt），再喂 Sana |
| `outputs/openuni_b512` | OpenUni-B-512（InternVL3-1B + Sana-0.6B-512px + 256 meta-queries） | 官方 SFT ckpt `blip3o60k`，代码取自 OpenUni geneval worktree（含 diffusers 0.32.0 tokenizer fix） |

## 用法

```bash
# 全部模型（默认 umbrella_baseball）
bash probes/scripts/run_all.sh

# 单个模型
CUDA_VISIBLE_DEVICES=1 /home/mingjun/.conda/envs/bagel/bin/python \
    probes/scripts/gen_sana512.py --out probes/outputs/sana512 --gpu 1
```

## 新增 prompt

在 `prompts/` 下加一行（或新建 jsonl 文件）：

```json
{"id": "my_probe", "prompt": "a red panda sitting on a wooden board", "note": "..."}
```

所有脚本都读 `--prompts` 指向的 jsonl，无需改脚本。若要按 GenEval 语义校验增强
结果，把 `gen_sana512_ivl3enh.py` 的 `required_check` 换成完整校验器。

## 新增模型

在 `scripts/` 加一个生成脚本，输出布局与其它模型一致（`outputs/<model>/<prompt_id>/`
下 `<seed>.png` + `meta.jsonl`），并把 meta 行的 `model` 字段填好。

## 运行环境

`bagel` conda env（torch 2.5.1+cu124 / diffusers 0.32.0 / transformers 4.49.0 /
flash_attn 2.7.4）。模型路径均为本地：`~/models/sana-600m-512px`、
`~/models/internvl3-1b`、`/data/mingjun/openuni_model/...blip3o60k.pth`。

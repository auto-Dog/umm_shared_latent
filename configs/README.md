# Configs

Configs compose by inheritance: each file's `extends:` is resolved **relative to
its own directory**, and child values deep-merge over the parent. Click through
the chain rather than duplicating keys.

## Live entry points

| Entry point | Config |
|---|---|
| `scripts/run_local_ivl3_cfg10.sh` | `local_pt_ivl3_cfg10.yaml` — main local single-card run |
| `scripts/run_remote.sh` | `remote_pt_cc12m_1_2m.yaml` → `remote_pt_edit_1_2m.yaml` → `remote_pt_finetune_blip3o_metaquery.yaml` (3-stage remote pipeline) |
| `scripts/run_local.sh`, `scripts/sanity.py` | `local_pt_smoke.yaml` — 600-sample runtime smoke |
| `scripts/run_staged_pt.py --base-config` | `local_pt_ivl3.yaml` default |

## Shared bases (required by the above)

```
baseline_pt.yaml
└─ local_pt.yaml                     # local-weight runtime profile
   ├─ local_pt_smoke.yaml
   └─ local_pt_ivl3.yaml             # InternVL3-1B backbone
      └─ local_pt_ivl3_cfg10.yaml    # OpenUni-aligned run profile
         ├─ remote_pt_cc12m_1_2m.yaml
         ├─ remote_pt_edit_1_2m.yaml
         └─ remote_pt_finetune_blip3o_metaquery.yaml
```

`deepspeed_zero1.json` is referenced by `baseline_pt.yaml`.

## Legacy (archived, not wired to any script)

- `legacy/baseline_dm.yaml` and `legacy/ablations/*` — the first-round ablation
  matrix described in [`docs/ablations.md`](../docs/ablations.md). Kept for
  reference; nothing currently runs them.

# Attribution

The model assembly, MetaQuery-token vocabulary construction, Qwen2.5-VL forward
path, query-span extraction, prompt format, VAE handling, and flow-matching
baseline in this project are adapted from the adjacent `metaquery/` reference
implementation by Meta Platforms, Inc.

The referenced implementation is distributed under CC BY-NC 4.0. See
`../metaquery/LICENSE` and `../metaquery/README.md` for its license and upstream
attribution. UniQuery-specific changes include the light connector, streaming
data mixture, optional loss ablations, adapter checkpoints, and resilient
training facilities.

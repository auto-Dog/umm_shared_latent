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

The `src/umm_uniquery/modeling/internvl3/` directory contains the InternVL3
model sources (InternVLChatModel, InternVisionModel, and related
configs/conversation helpers) copied verbatim from the adjacent `OpenUni/`
reference, Copyright (c) 2024 OpenGVLab, distributed under The MIT License (see
the header comment in each file). The `UniQueryInternVL3Model` wrapper in
`modeling/internvl3_model.py` is this project's own code; it drives InternVL3's
frozen Llama language model through the same special-token MetaQuery mechanism
as the Qwen backbone instead of the OpenUni independent meta_queries parameter.

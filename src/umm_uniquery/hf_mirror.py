"""Rewrite Hub streaming URLs through a configured Hugging Face mirror.

Some Hub API responses contain absolute ``https://huggingface.co/...`` URLs
even when ``HF_ENDPOINT`` points at a mirror. On hosts that cannot reach the
default endpoint, streaming then hangs while following those URLs. This module
patches the two URL hand-off points used by datasets/huggingface_hub. It is a
no-op for the default endpoint.
"""

from __future__ import annotations

import os


_HF_PREFIX = "https://huggingface.co"
_PATCH_MARKER = "_uniquery_hf_mirror_patched"


def _rewrite(url: str, endpoint: str) -> str:
    if isinstance(url, str) and url.startswith(_HF_PREFIX):
        return endpoint + url[len(_HF_PREFIX) :]
    return url


def install_hf_mirror_rewrite() -> None:
    """Install idempotent mirror rewrites when ``HF_ENDPOINT`` is non-default."""

    endpoint = os.environ.get("HF_ENDPOINT", _HF_PREFIX).rstrip("/")
    if _HF_PREFIX in endpoint:
        return

    # datasets/fsspec opens exported parquet and WebDataset URLs verbatim.
    try:
        from datasets.utils import file_utils
    except ImportError:
        file_utils = None
    if file_utils is not None and not getattr(file_utils, _PATCH_MARKER, False):
        original_prepare = file_utils._prepare_single_hop_path_and_storage_options

        def patched_prepare(urlpath, download_config=None):
            return original_prepare(_rewrite(urlpath, endpoint), download_config)

        file_utils._prepare_single_hop_path_and_storage_options = patched_prepare
        setattr(file_utils, _PATCH_MARKER, True)

    # Pagination Link headers can also point back to the default Hub host.
    try:
        import huggingface_hub.utils._pagination as pagination
    except ImportError:
        pagination = None
    if pagination is not None and not getattr(pagination, _PATCH_MARKER, False):
        original_backoff = pagination.http_backoff

        def patched_backoff(method, url, *args, **kwargs):
            return original_backoff(method, _rewrite(url, endpoint), *args, **kwargs)

        pagination.http_backoff = patched_backoff
        setattr(pagination, _PATCH_MARKER, True)

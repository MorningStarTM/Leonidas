"""Fetch the SMPL-X model file on a fresh deployment (e.g. Streamlit
Community Cloud) where it can't be committed to the repo.

The model is registration-gated at https://smpl-x.is.tue.mpg.de and this
project never redistributes it (see src/smplx/config.py) — this module only
downloads *your own* privately-hosted copy, using a Google Drive file id you
supply, straight into the same `src/model/` location `find_smplx_model()`
already searches. Local development is unaffected: if the file is already
there (or LEONIDAS_SMPLX_MODEL already points at one), this is a no-op.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TARGET = _REPO_ROOT / "src" / "model" / "SMPLX_FEMALE.npz"
_SECRET_KEY = "SMPLX_MODEL_DRIVE_ID"


def _get_drive_id() -> Optional[str]:
    env_value = os.environ.get(_SECRET_KEY)
    if env_value:
        return env_value
    try:
        import streamlit as st
        if _SECRET_KEY in st.secrets:
            return st.secrets[_SECRET_KEY]
    except Exception:
        pass
    return None


def ensure_smplx_model_downloaded() -> str:
    """Download the model into `src/model/` if it isn't already present.

    Safe to call unconditionally at app startup: does nothing when a model
    file already exists locally or no drive id is configured, so
    `find_smplx_model()` / `SMPLXModelNotFoundError` still handle every
    other case exactly as before.

    Returns a short, human-readable status string so the caller can surface
    *why* no download happened (secret missing vs. already present vs. a
    real download failure) instead of only seeing the generic "no model
    found" error two layers away.
    """
    from src.smplx.config import find_smplx_model

    if find_smplx_model() is not None:
        return "model already present locally, skipped download"

    drive_id = _get_drive_id()
    if not drive_id:
        return (f"no local model found and no '{_SECRET_KEY}' secret/env var is set, "
                "so nothing was downloaded")

    import gdown

    _TARGET.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=drive_id, output=str(_TARGET), quiet=False)
    if result is None or not _TARGET.is_file():
        raise RuntimeError(
            f"gdown could not download drive file id '{drive_id}' to {_TARGET} — "
            "check the file is shared as \"Anyone with the link\" and the id is correct."
        )
    return f"downloaded model to {_TARGET} ({_TARGET.stat().st_size} bytes)"

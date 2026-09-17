import pytest

from src.smplx.config import find_smplx_model


@pytest.fixture(scope="session")
def smplx_model_path():
    path = find_smplx_model()
    if path is None:
        pytest.skip(
            "No SMPL-X model file found (registration-gated asset; set "
            "LEONIDAS_SMPLX_MODEL to enable model-dependent tests)."
        )
    return str(path)


@pytest.fixture(scope="session")
def smplx_body(smplx_model_path):
    from src.smplx.fitting.body import SMPLXBody
    return SMPLXBody(model_path=smplx_model_path)

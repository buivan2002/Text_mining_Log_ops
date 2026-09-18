import pytest

from src.pipeline import layer4_validation


def test_validate_not_implemented():
    with pytest.raises(NotImplementedError):
        layer4_validation.validate([])

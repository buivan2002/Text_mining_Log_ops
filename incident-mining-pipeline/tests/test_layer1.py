import pytest

from src.pipeline import layer1_preprocessing


def test_preprocess_not_implemented():
    with pytest.raises(NotImplementedError):
        layer1_preprocessing.preprocess("sample raw log text")

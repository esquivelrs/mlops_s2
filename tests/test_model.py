import pytest
import torch

from tests import _PATH_PACKAGE, _PATH_MODELS
# add folder to import
import os
import sys
sys.path.append(_PATH_PACKAGE)

from models.model import MyNeuralNet


def test_model():
    model = MyNeuralNet
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)
    
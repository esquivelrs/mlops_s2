
import pytest

from tests import _PATH_DATA
import torch
import os


# def test_data():
#     dataset = MNIST(...)
#     assert len(dataset) == N_train for training and N_test for test
#     assert that each datapoint has shape [1,28,28] or [784] depending on how you choose to format
#     assert that all labels are represented


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    dataset = torch.load(os.path.join(_PATH_DATA, "processed/train_data.pt") )
    assert len(dataset) == 25000
    assert dataset[0][0].shape == (1,28,28)
    assert dataset[0][1] == 5
    assert dataset[24999][0].shape == (1,28,28)
    assert dataset[24999][1] == 9

    dataset = torch.load(os.path.join(_PATH_DATA, "processed/test_data.pt") )
    assert len(dataset) == 5000
    assert dataset[0][0].shape == (1,28,28)
    assert dataset[0][1] == 7
    assert dataset[4999][0].shape == (1,28,28)
    assert dataset[4999][1] == 0
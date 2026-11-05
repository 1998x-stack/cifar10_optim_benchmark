"""Regression tests for deterministic CIFAR-10 splits and transform isolation."""
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Subset

from experiments.run_single import _get_cifar_loaders


class FakeCIFAR10:
    calls = []

    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.calls.append(self)

    def __len__(self):
        return 20 if self.train else 5

    def __getitem__(self, index):
        return index, 0


class DataSplitTests(unittest.TestCase):
    def setUp(self):
        FakeCIFAR10.calls = []
        self.config = {
            "data": {"train_val_split": 0.75, "val_seed": 123, "batch_size": 4,
                     "test_batch_size": 4, "download": True},
            "experiment": {"num_workers": 0},
        }

    def test_validation_does_not_replace_training_augmentation(self):
        with patch("experiments.run_single.datasets.CIFAR10", FakeCIFAR10):
            train, val, test = _get_cifar_loaders(self.config, torch.device("cpu"))

        self.assertIsInstance(train.dataset, Subset)
        self.assertIsInstance(val.dataset, Subset)
        self.assertIsNot(train.dataset.dataset, val.dataset.dataset)
        self.assertIsNot(train.dataset.dataset.transform, val.dataset.dataset.transform)
        self.assertEqual(len(train.dataset), 15)
        self.assertEqual(len(val.dataset), 5)
        self.assertEqual(set(train.dataset.indices) & set(val.dataset.indices), set())
        self.assertEqual(sorted(train.dataset.indices + val.dataset.indices), list(range(20)))
        self.assertEqual(len(test.dataset), 5)
        self.assertEqual([d.download for d in FakeCIFAR10.calls], [True, False, True])

    def test_split_is_repeatable_for_same_seed(self):
        with patch("experiments.run_single.datasets.CIFAR10", FakeCIFAR10):
            train1, val1, _ = _get_cifar_loaders(self.config, torch.device("cpu"))
            train2, val2, _ = _get_cifar_loaders(self.config, torch.device("cpu"))
        self.assertEqual(train1.dataset.indices, train2.dataset.indices)
        self.assertEqual(val1.dataset.indices, val2.dataset.indices)


if __name__ == "__main__":
    unittest.main()

"""
Stanford Cars Dataset loader.

Download: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
           OR official: http://ai.stanford.edu/~jkrause/cars/car_dataset.html

Expected layout (after kaggle download):
    <root>/
        cars_train/          ← training images
        cars_test/           ← test images
        devkit/
            cars_train_annos.mat
            cars_test_annos_withlabels.mat   (or cars_test_annos.mat)
            cars_meta.mat

scipy.io is used to read the .mat annotation files.
"""

from collections import defaultdict
from pathlib import Path

import scipy.io

from .base import FGVCDataset


class StanfordCarsDataset(FGVCDataset):

    name = "cars"

    def load(self):
        devkit = self.root / "devkit"
        meta   = scipy.io.loadmat(devkit / "cars_meta.mat")
        class_names = [
            c[0] for c in meta["class_names"][0]
        ]  # list of 196 strings

        train_dict: dict[int, list[Path]] = defaultdict(list)
        test_dict:  dict[int, list[Path]] = defaultdict(list)

        # ── Training annotations ──────────────────────────────────────────────
        train_annos = scipy.io.loadmat(devkit / "cars_train_annos.mat")
        for anno in train_annos["annotations"][0]:
            fname = str(anno["fname"][0])
            label = int(anno["class"][0][0]) - 1  # 0-indexed
            train_dict[label].append(self.root / "cars_train" / fname)

        # ── Test annotations ──────────────────────────────────────────────────
        test_mat = devkit / "cars_test_annos_withlabels.mat"
        if not test_mat.exists():
            test_mat = devkit / "cars_test_annos.mat"
        test_annos = scipy.io.loadmat(test_mat)
        for anno in test_annos["annotations"][0]:
            fname = str(anno["fname"][0])
            label = int(anno["class"][0][0]) - 1
            test_dict[label].append(self.root / "cars_test" / fname)

        return class_names, dict(train_dict), dict(test_dict)

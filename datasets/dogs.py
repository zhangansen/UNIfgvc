"""
Stanford Dogs Dataset loader.

Download: http://vision.stanford.edu/aditya86/ImageNetDogs/
          OR https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

Expected layout:
    <root>/
        Images/
            n02085620-Chihuahua/
            n02085782-Japanese_spaniel/
            ...
        lists/
            train_list.mat
            test_list.mat

120 dog breed classes.
"""

from collections import defaultdict
from pathlib import Path

import scipy.io

from .base import FGVCDataset


class StanfordDogsDataset(FGVCDataset):

    name = "dogs"

    def load(self):
        lists_dir  = self.root / "lists"
        images_dir = self.root / "Images"

        train_mat = scipy.io.loadmat(lists_dir / "train_list.mat")
        test_mat  = scipy.io.loadmat(lists_dir / "test_list.mat")

        # class_names from folder names (sorted for stability)
        folders = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
        folder2idx = {f: i for i, f in enumerate(folders)}
        class_names = [
            f.split("-", 1)[-1].replace("_", " ") for f in folders
        ]

        train_dict: dict[int, list[Path]] = defaultdict(list)
        test_dict:  dict[int, list[Path]] = defaultdict(list)

        def _fill(mat, target):
            for entry in mat["file_list"].flatten():
                rel = str(entry[0])          # e.g. "n02085620-Chihuahua/image.jpg"
                folder = rel.split("/")[0]
                label  = folder2idx[folder]
                target[label].append(images_dir / rel)

        _fill(train_mat, train_dict)
        _fill(test_mat,  test_dict)

        return class_names, dict(train_dict), dict(test_dict)

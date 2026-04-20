"""
Food-101 Dataset loader.

Download: https://www.kaggle.com/datasets/dansbecker/food-101
          OR https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

Expected layout:
    <root>/food-101/
        images/
            apple_pie/
            baby_back_ribs/
            ...
        meta/
            classes.txt
            train.txt
            test.txt

101 food categories.
"""

from collections import defaultdict
from pathlib import Path

from .base import FGVCDataset


class Food101Dataset(FGVCDataset):

    name = "food101"

    def __init__(self, root: str | Path):
        super().__init__(root)
        self._food = self.root / "food-101"

    def load(self):
        meta_dir   = self._food / "meta"
        images_dir = self._food / "images"

        with open(meta_dir / "classes.txt") as f:
            classes = [line.strip() for line in f]

        class2idx   = {c: i for i, c in enumerate(classes)}
        class_names = [c.replace("_", " ") for c in classes]

        train_dict: dict[int, list[Path]] = defaultdict(list)
        test_dict:  dict[int, list[Path]] = defaultdict(list)

        def _fill(txt_file: Path, target: dict):
            with open(txt_file) as f:
                for line in f:
                    rel = line.strip()          # e.g. "apple_pie/1005649"
                    cls = rel.split("/")[0]
                    label = class2idx[cls]
                    target[label].append(images_dir / f"{rel}.jpg")

        _fill(meta_dir / "train.txt", train_dict)
        _fill(meta_dir / "test.txt",  test_dict)

        return class_names, dict(train_dict), dict(test_dict)

"""
FGVC-Aircraft Dataset loader.

Download: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

Expected layout:
    <root>/fgvc-aircraft-2013b/data/
        images/
        images_variant_train.txt
        images_variant_val.txt
        images_variant_test.txt
        variants.txt

We use the 'variant' level (100 classes) which is the standard benchmark.
Train = train + val splits.
"""

from collections import defaultdict
from pathlib import Path

from .base import FGVCDataset


class AircraftDataset(FGVCDataset):

    name = "aircraft"

    def __init__(self, root: str | Path):
        super().__init__(root)
        self._data = self.root / "fgvc-aircraft-2013b" / "data"

    def load(self):
        # ── Class names ───────────────────────────────────────────────────────
        variants_file = self._data / "variants.txt"
        class_names: list[str] = []
        variant2idx: dict[str, int] = {}
        with open(variants_file) as f:
            for i, line in enumerate(f):
                name = line.strip()
                class_names.append(name)
                variant2idx[name] = i

        train_dict: dict[int, list[Path]] = defaultdict(list)
        test_dict:  dict[int, list[Path]] = defaultdict(list)

        def _read_split(fname: str, target: dict):
            path = self._data / fname
            with open(path) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    img_id, variant = parts[0], parts[1]
                    label = variant2idx[variant]
                    img_path = self._data / "images" / f"{img_id}.jpg"
                    target[label].append(img_path)

        _read_split("images_variant_train.txt", train_dict)
        _read_split("images_variant_val.txt",   train_dict)  # val → train
        _read_split("images_variant_test.txt",  test_dict)

        return class_names, dict(train_dict), dict(test_dict)

"""
Oxford-IIIT Pet Dataset loader.

Download: https://www.robots.ox.ac.uk/~vgg/data/pets/

Expected layout:
    <root>/
        images/          ← all images (*.jpg)
        annotations/
            trainval.txt
            test.txt

37 classes (25 dog breeds + 12 cat breeds).
Class label is encoded in the filename prefix.
"""

from collections import defaultdict
from pathlib import Path

from .base import FGVCDataset


class OxfordPetsDataset(FGVCDataset):

    name = "pets"

    def load(self):
        annotations = self.root / "annotations"
        images_dir  = self.root / "images"

        # Gather all class names (sorted unique prefixes)
        all_names: set[str] = set()

        def _parse(txt_file: Path):
            entries = []
            with open(txt_file) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    img_id = parts[0]           # e.g. "Abyssinian_100"
                    label  = int(parts[1]) - 1  # 1-indexed → 0-indexed
                    class_name = "_".join(img_id.split("_")[:-1]).replace("_", " ")
                    all_names.add(class_name)
                    entries.append((img_id, label, class_name))
            return entries

        train_entries = _parse(annotations / "trainval.txt")
        test_entries  = _parse(annotations / "test.txt")

        # Build a stable label→name mapping from label index
        # (labels in the file are already 1–37, consistent across splits)
        label2name: dict[int, str] = {}
        for img_id, label, cname in train_entries + test_entries:
            label2name[label] = cname

        n_classes   = max(label2name) + 1
        class_names = [label2name.get(i, f"class_{i}") for i in range(n_classes)]

        train_dict: dict[int, list[Path]] = defaultdict(list)
        test_dict:  dict[int, list[Path]] = defaultdict(list)

        for img_id, label, _ in train_entries:
            train_dict[label].append(images_dir / f"{img_id}.jpg")
        for img_id, label, _ in test_entries:
            test_dict[label].append(images_dir / f"{img_id}.jpg")

        return class_names, dict(train_dict), dict(test_dict)

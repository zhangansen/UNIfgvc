from collections import defaultdict
from pathlib import Path

from .base import FGVCDataset


class CUBDataset(FGVCDataset):
    """
    CUB-200-2011.
    Download: https://www.kaggle.com/datasets/wenewone/cub2002011
    Expected layout:
        <root>/CUB_200_2011/
            images.txt
            image_class_labels.txt
            train_test_split.txt
            classes.txt
            images/
    """

    name = "cub"

    def __init__(self, root: str | Path, n_classes: int = 200):
        super().__init__(root)
        self.n_classes = n_classes
        self._cub = self.root / "CUB_200_2011"

    def load(self):
        img_id2path, img_id2label, img_id2split, class_names = {}, {}, {}, {}

        with open(self._cub / "images.txt") as f:
            for line in f:
                iid, path = line.strip().split()
                img_id2path[iid] = self._cub / "images" / path

        with open(self._cub / "image_class_labels.txt") as f:
            for line in f:
                iid, lbl = line.strip().split()
                img_id2label[iid] = int(lbl) - 1

        with open(self._cub / "train_test_split.txt") as f:
            for line in f:
                iid, s = line.strip().split()
                img_id2split[iid] = int(s)

        with open(self._cub / "classes.txt") as f:
            for line in f:
                cidx, cname = line.strip().split(" ", 1)
                class_names[int(cidx) - 1] = (
                    cname.split(".", 1)[-1].replace("_", " ")
                )

        train_dict: dict[int, list[Path]] = defaultdict(list)
        test_dict:  dict[int, list[Path]] = defaultdict(list)

        for iid, path in img_id2path.items():
            lbl = img_id2label[iid]
            if lbl >= self.n_classes:
                continue
            if img_id2split[iid] == 1:
                train_dict[lbl].append(path)
            else:
                test_dict[lbl].append(path)

        names = [class_names[i] for i in range(self.n_classes)]
        return names, dict(train_dict), dict(test_dict)

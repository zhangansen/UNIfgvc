from abc import ABC, abstractmethod
from pathlib import Path


class FGVCDataset(ABC):
    """
    Abstract base for all FGVC datasets.

    Subclasses must implement:
        load() -> (class_names, train_dict, test_dict)
    where
        class_names : list[str]           — human-readable class names
        train_dict  : dict[int, list[Path]] — label → train image paths
        test_dict   : dict[int, list[Path]] — label → test  image paths
    """

    name: str = "base"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @abstractmethod
    def load(self) -> tuple[list[str], dict[int, list[Path]], dict[int, list[Path]]]:
        ...

from .cub      import CUBDataset
from .cars     import StanfordCarsDataset
from .aircraft import AircraftDataset
from .pets     import OxfordPetsDataset
from .dogs     import StanfordDogsDataset
from .food101  import Food101Dataset

DATASET_REGISTRY = {
    "cub":      CUBDataset,
    "cars":     StanfordCarsDataset,
    "aircraft": AircraftDataset,
    "pets":     OxfordPetsDataset,
    "dogs":     StanfordDogsDataset,
    "food101":  Food101Dataset,
}


def get_dataset(name: str, root: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](root, **kwargs)

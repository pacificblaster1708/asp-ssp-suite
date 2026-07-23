from .synthetic import SyntheticPointDataset, CLASSES as SYNTHETIC_CLASSES
from . import corruptions

__all__ = ["SyntheticPointDataset", "SYNTHETIC_CLASSES", "corruptions",
           "build_dataset"]


def build_dataset(name: str, split: str, cfg: dict):
    """Factory used by experiments/run.py. Heavy deps imported lazily."""
    root = cfg.get("data_root", "./data")
    corr, sev = cfg.get("_corruption_fn"), cfg.get("_severity", 0)
    if name == "synthetic":
        return SyntheticPointDataset(
            n_per_class=cfg.get("n_per_class", 120 if split == "train" else 30),
            n_points=cfg.get("n_points", 256), k_slices=cfg.get("k_slices", 16),
            points_per_slice=cfg.get("points_per_slice", 32),
            seed=0 if split == "train" else 1, corruption=corr, severity=sev)
    if name in {"modelnet40", "modelnet10"}:
        from .modelnet import ModelNetDataset
        return ModelNetDataset(root, int(name[-2:].lstrip("t") or 40)
                               if name != "modelnet10" else 10, split,
                               cfg.get("n_points", 1024), cfg.get("k_slices", 16),
                               cfg.get("points_per_slice", 128),
                               corruption=corr, severity=sev)
    if name == "scanobjectnn":
        from .scanobjectnn import ScanObjectNNDataset
        return ScanObjectNNDataset(root, split, cfg.get("n_points", 1024),
                                   cfg.get("k_slices", 16),
                                   cfg.get("points_per_slice", 128),
                                   corruption=corr, severity=sev)
    if name in {"cifar10", "cifar100"}:
        from .cifar_patch import CIFARPatchDataset
        return CIFARPatchDataset(root, int(name[5:]), split,
                                 corruption=corr, severity=sev,
                                 limit=cfg.get("limit"))
    if name in {"nmnist","dvscifar10","dvsgesture","shd","ssc","ncaltech101"}:
        from .neuromorphic import NeuromorphicPointDataset
        return NeuromorphicPointDataset(name, root, split,
            n_points=cfg.get("n_points",1024), k_slices=cfg.get("k_slices",16),
            points_per_slice=cfg.get("points_per_slice",64),
            event_cap=cfg.get("event_cap",2048), cache=cfg.get("cache",True),
            limit=cfg.get("limit"), corruption=corr, severity=sev)
    raise ValueError(f"unknown dataset {name}")

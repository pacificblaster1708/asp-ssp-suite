
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ..geometry import normalize_cloud, slice_point_cloud

NUM_CLASSES = {"nmnist":10,"dvscifar10":10,"dvsgesture":11,"shd":20,"ssc":35,"ncaltech101":101}

def _split_indices(n, train, frac_test=0.1, seed=1234):
    perm = np.random.default_rng(seed).permutation(n)
    n_test = max(1, int(frac_test * n))
    return (perm[n_test:] if train else perm[:n_test]).tolist()

def _build_base(name, root, split):
    import tonic
    D, train = tonic.datasets, split == "train"
    if name == "nmnist":      return D.NMNIST(save_to=root, train=train), None
    if name == "dvsgesture":  return D.DVSGesture(save_to=root, train=train), None
    if name == "shd":         return D.SHD(save_to=root, train=train), None
    if name == "ssc":         return D.SSC(save_to=root, split="train" if train else "test"), None
    if name == "dvscifar10":
        b = D.CIFAR10DVS(save_to=root); return b, _split_indices(len(b), train)
    if name == "ncaltech101":
        b = D.NCALTECH101(save_to=root); return b, _split_indices(len(b), train)
    raise ValueError(name)

def _events_to_cloud(ev, cap):
    names = ev.dtype.names
    t = np.asarray(ev["t"], "float64"); x = np.asarray(ev["x"], "float64")
    y = np.asarray(ev["y"], "float64") if "y" in names else np.zeros_like(x)
    m = len(t)
    if m > cap:
        s = np.linspace(0, m - 1, cap).astype("int64"); x, y, t = x[s], y[s], t[s]
    def nrm(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) * 2 - 1 if hi > lo else np.zeros_like(a)
    return np.stack([nrm(x), nrm(y), nrm(t)], -1).astype("float32")

class NeuromorphicPointDataset(Dataset):
    def __init__(self, name, root="./data", split="train", n_points=1024, k_slices=16,
                 points_per_slice=64, event_cap=2048, cache=True, limit=None,
                 augment=None, corruption=None, severity=0):
        self.name, self.split = name, split
        self.n_points, self.k, self.p = n_points, k_slices, points_per_slice
        self.event_cap, self.cache = event_cap, cache
        self.corruption, self.severity = corruption, severity
        self.num_classes = NUM_CLASSES[name]
        self.augment = (split == "train") if augment is None else augment
        self.base, self.idx = _build_base(name, root, split)
        self._n = len(self.idx) if self.idx is not None else len(self.base)
        if limit: self._n = min(self._n, int(limit))
        self.cache_dir = os.path.join(root, "neuro_cache", f"{name}_{split}")
        if cache: os.makedirs(self.cache_dir, exist_ok=True)
    def __len__(self): return self._n
    def _bi(self, i): return self.idx[i] if self.idx is not None else i
    def _load(self, i):
        cp = os.path.join(self.cache_dir, f"{i}.npy")
        if self.cache and os.path.exists(cp):
            a = np.load(cp); return a[:, :3].astype("float32"), int(a[0, 3])
        ev, tgt = self.base[self._bi(i)]
        c = _events_to_cloud(ev, self.event_cap); lab = int(tgt)
        if self.cache:
            tmp = cp + ".tmp"
            np.save(tmp, np.concatenate([c, np.full((len(c), 1), lab, "float32")], 1).astype("float16"))
            os.replace(tmp + ".npy" if os.path.exists(tmp + ".npy") else tmp, cp)
        return c, lab
    def __getitem__(self, i):
        c, lab = self._load(i); m = len(c)
        g = np.random.default_rng() if self.augment else np.random.default_rng(i * 7919 + 13)
        sel = g.choice(m, self.n_points, replace=(m < self.n_points))
        cloud = torch.from_numpy(c[sel]).float()
        if self.augment: cloud = cloud + torch.randn_like(cloud) * 0.01
        cloud = normalize_cloud(cloud.unsqueeze(0)).squeeze(0)
        if self.corruption is not None and self.severity > 0:
            tg = torch.Generator().manual_seed(i * 2741 + 7)
            cloud = normalize_cloud(self.corruption(cloud, self.severity, tg).unsqueeze(0)).squeeze(0)
        sl, d, a = slice_point_cloud(cloud.unsqueeze(0), self.k, self.p)
        return sl.squeeze(0), d.squeeze(0), a.squeeze(0), lab

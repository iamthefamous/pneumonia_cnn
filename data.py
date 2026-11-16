import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).parent
IMG_SIZE = 128
CLASSES = ["normal", "pneumonia"]
DATA_FILE = ROOT / "data" / f"pneumoniamnist_{IMG_SIZE}.npz"
DATA_URL = f"https://zenodo.org/records/10519652/files/{DATA_FILE.name}?download=1"
DATA_MD5 = "05b46931834c231683c68f40c47b2971"
SAMPLES_DIR = ROOT / "samples"

AUGMENT = v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))


def md5(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def download():
    if DATA_FILE.exists() and md5(DATA_FILE) == DATA_MD5:
        return
    DATA_FILE.parent.mkdir(exist_ok=True)
    print(f"Downloading {DATA_FILE.name} ...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    if md5(DATA_FILE) != DATA_MD5:
        DATA_FILE.unlink()
        raise RuntimeError(f"Checksum mismatch for {DATA_FILE.name}, run the download again")


def load_split(split):
    if not DATA_FILE.exists():
        download()
    with np.load(DATA_FILE) as data:
        return data[f"{split}_images"], data[f"{split}_labels"].ravel()


class XRayDataset(Dataset):
    def __init__(self, split, augment=False):
        images, labels = load_split(split)
        self.images = torch.from_numpy(images).unsqueeze(1).float() / 255
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i]
        if self.augment:
            image = AUGMENT(image)
        return image, self.labels[i]


def preprocess_image(path):
    image = ImageOps.fit(Image.open(path).convert("L"), (IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).unsqueeze(0)


def export_samples(per_class=5):
    images, labels = load_split("test")
    SAMPLES_DIR.mkdir(exist_ok=True)
    for label, name in enumerate(CLASSES):
        for i in np.flatnonzero(labels == label)[:per_class]:
            Image.fromarray(images[i]).save(SAMPLES_DIR / f"test{i:03d}_{name}.png")


def describe():
    for split in ("train", "val", "test"):
        _, labels = load_split(split)
        counts = ", ".join(f"{name} {np.sum(labels == i)}" for i, name in enumerate(CLASSES))
        print(f"{split:<6}{len(labels):>5} images  ({counts})")


if __name__ == "__main__":
    download()
    describe()
    export_samples()

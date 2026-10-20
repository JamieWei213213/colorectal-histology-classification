"""Official splits, train-only sampling, and explicit model-specific transforms."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import Dataset
from .core import save_json

CLASS_NAMES = ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
               "normal colon mucosa", "cancer-associated stroma", "colorectal adenocarcinoma epithelium"]
EXPECTED_SIZES = {"train": 89996, "val": 10004, "test": 7180}


def sample_training_indices(labels, count, seed):
    labels = np.asarray(labels).reshape(-1)
    if count is None or count == len(labels):
        return np.arange(len(labels))
    if not 9 <= count < len(labels):
        raise ValueError("Training subset must be between 9 and the official training size")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=count, random_state=seed)
    chosen, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(chosen)


def official_dataset(split, root):
    # Lazy import lets the core smoke test run without network-only dependencies.
    from medmnist import INFO, PathMNIST
    if split not in EXPECTED_SIZES:
        raise ValueError("Use an official train, val, or test split")
    info = INFO["pathmnist"]
    assert [info["label"][str(i)] for i in range(9)] == CLASS_NAMES
    assert info["n_samples"] == EXPECTED_SIZES
    Path(root).mkdir(parents=True, exist_ok=True)
    data = PathMNIST(split=split, root=str(root), size=28, download=True)
    assert len(data) == EXPECTED_SIZES[split]
    assert data.imgs.shape == (len(data), 28, 28, 3)
    assert data.imgs.dtype == np.uint8
    assert np.isin(data.labels, np.arange(9)).all()
    return data


class PatchDataset(Dataset):
    def __init__(self, images, labels, transform, indices=None):
        self.images, self.labels, self.transform = images, np.asarray(labels).reshape(-1), transform
        self.indices = np.arange(len(images)) if indices is None else np.asarray(indices)
        if len(self.images) != len(self.labels):
            raise ValueError("Mismatched images and labels")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = int(self.indices[index])
        image = Image.fromarray(self.images[source_index]).convert("RGB")
        return self.transform(image), int(self.labels[source_index])


def transforms_for(model_name, training=False):
    from torchvision import transforms as T
    from torchvision.models import ResNet18_Weights
    augmentation = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()] if training else []
    if model_name == "cnn":
        return T.Compose(augmentation + [T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    if model_name == "resnet18":
        # Exact official pretrained inference preset: bilinear 256 -> center crop 224,
        # float [0,1], ImageNet mean/std. Only train has random flips beforehand.
        return T.Compose(augmentation + [ResNet18_Weights.IMAGENET1K_V1.transforms()])
    raise ValueError(model_name)


def build_model(name, pretrained=True):
    from .core import CompactCNN
    if name == "cnn":
        return CompactCNN()
    if name != "resnet18":
        raise ValueError(name)
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 9)
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.freeze_early_layers = True
    return model


def explore(train, val, indices, output):
    import matplotlib.pyplot as plt
    from medmnist import INFO
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"official_train": train.labels.reshape(-1), "official_val": val.labels.reshape(-1),
              "used_train": train.labels.reshape(-1)[indices]}
    counts = pd.DataFrame({split: np.bincount(labels, minlength=9) for split, labels in arrays.items()}, index=CLASS_NAMES)
    counts.index.name = "class"
    counts.to_csv(output / "class_counts_before_test.csv")
    ax = counts.plot.bar(figsize=(12, 5))
    ax.set_ylabel("Image patches")
    ax.set_title("Official train/validation and selected training subset; test remains sealed")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "class_counts.png", dpi=160)
    plt.close(ax.figure)
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for cls, ax in enumerate(axes.flat):
        index = int(np.flatnonzero(train.labels.reshape(-1) == cls)[0])
        ax.imshow(train.imgs[index], interpolation="nearest")
        ax.set_title(f"{cls}: {CLASS_NAMES[cls]}\ntrain index {index}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Official training examples — 28 x 28 pixels; enlarged only for display")
    fig.tight_layout()
    fig.savefig(output / "labeled_training_examples.png", dpi=160)
    plt.close(fig)
    np.save(output / "training_indices.npy", indices)
    save_json(output / "dataset_info.json", INFO["pathmnist"])
    return counts

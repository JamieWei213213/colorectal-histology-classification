"""Orchestrate a locked train/validation protocol followed by final test evaluation."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from .core import environment, fit_model, make_loader, pass_batches, save_json, seed_everything, sha256, write_evaluation
from .data import CLASS_NAMES, PatchDataset, build_model, explore, official_dataset, sample_training_indices, transforms_for


@dataclass
class Config:
    profile: str = "today"
    seed: int = 42
    train_size: int | None = 20000
    batch_size: int = 64
    workers: int = 0  # portable across Windows and Colab; avoids duplicating RAM arrays
    cnn_epochs: int = 6
    resnet_epochs: int = 4
    patience: int = 2
    cnn_lr: float = 1e-3
    resnet_lr: float = 3e-4

    @classmethod
    def for_profile(cls, profile):
        if profile == "today":
            return cls()
        if profile == "full":
            return cls(profile="full", train_size=None, cnn_epochs=8, resnet_epochs=6)
        raise ValueError("Choose today or full. Synthetic smoke tests are separate.")


class Experiment:
    def __init__(self, config, root="data", output="outputs/experiment"):
        self.config, self.root, self.output = config, Path(root), Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError("Choose an empty output directory to preserve prior results.")
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        seed_everything(config.seed)
        self.config_snapshot = asdict(config)
        save_json(self.output / "config.json", self.config_snapshot)
        save_json(self.output / "environment.json", environment())
        (self.output / "requirements-resolved.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
        source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
        save_json(self.output / "source_hashes.json", source_hashes)
        self.summaries = {}
        self.train = self.val = None
        self.selection = None
        print(f"Device={self.device}; profile={config.profile}; output={self.output}")

    def _check_config(self):
        if asdict(self.config) != self.config_snapshot:
            raise RuntimeError("Configuration changed after initialization; create a fresh run before testing.")

    def prepare(self):
        self._check_config()
        if self.train is not None:
            return pd.read_csv(self.output / "exploration/class_counts_before_test.csv", index_col=0)
        self.train = official_dataset("train", self.root)
        self.val = official_dataset("val", self.root)
        self.indices = sample_training_indices(self.train.labels, self.config.train_size, self.config.seed)
        counts = explore(self.train, self.val, self.indices, self.output / "exploration")
        save_json(self.output / "data_manifest.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(), "source_size": 28,
            "train_used": len(self.indices), "official_train": len(self.train), "validation_used": len(self.val),
            "train_indices_sha256": sha256(self.output / "exploration/training_indices.npy"),
            "npz_sha256": sha256(self.root / "pathmnist.npz"),
            "test_accessed_for_analysis": False,
            "note": "Official NPZ physically contains all three splits. Test arrays are not inspected by this workflow before selection is frozen.",
        })
        return counts

    def _loader(self, name, data, training=False, indices=None):
        dataset = PatchDataset(data.imgs, data.labels, transforms_for(name, training), indices)
        return make_loader(dataset, self.config.batch_size, training, self.config.seed,
                           self.config.workers, self.device)

    def preflight(self):
        """Real training-data forward pass for each model, including pretrained download."""
        if self.train is None:
            raise RuntimeError("Run prepare first")
        checks = {}
        for name in ("cnn", "resnet18"):
            seed_everything(self.config.seed)
            model = build_model(name).to(self.device).eval()
            loader = self._loader(name, self.train, indices=self.indices[:2])
            x, y = next(iter(loader))
            with torch.no_grad():
                logits = model(x.to(self.device))
            assert logits.shape == (2, 9) and torch.isfinite(logits).all()
            checks[name] = {"input_shape": list(x.shape), "output_shape": list(logits.shape),
                            "labels_shape": list(y.shape), "pretrained": name == "resnet18"}
            del model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        save_json(self.output / "preflight.json", checks)
        return checks

    def train_one(self, name):
        self._check_config()
        if self.train is None:
            raise RuntimeError("Run prepare first")
        if self.selection is not None or (self.output / "selection.json").exists():
            raise RuntimeError("Selection is frozen; training after test preparation is prohibited in this run")
        if name in self.summaries:
            return self.summaries[name]
        if name not in ("cnn", "resnet18"):
            raise ValueError(name)
        seed_everything(self.config.seed)
        model = build_model(name).to(self.device)
        is_cnn = name == "cnn"
        metadata = {
            "model": name, "scope": f"PathMNIST {self.config.profile}; train n={len(self.indices)}",
            "config": self.config_snapshot, "pretrained": not is_cnn,
            "input_resolution": 28 if is_cnn else 224,
            "source_resolution": 28,
            "preprocessing": "RGB [0,1]; mean/std=0.5" if is_cnn else "ResNet18 IMAGENET1K_V1.transforms(): resize256/crop224/ImageNet normalization",
        }
        summary = fit_model(model, self._loader(name, self.train, True, self.indices),
                            self._loader(name, self.val), CLASS_NAMES, self.output / name,
                            self.device, self.config.cnn_epochs if is_cnn else self.config.resnet_epochs,
                            self.config.cnn_lr if is_cnn else self.config.resnet_lr,
                            self.config.patience, metadata)
        self.summaries[name] = summary
        del model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return summary

    def freeze_selection(self):
        self._check_config()
        if set(self.summaries) != {"cnn", "resnet18"}:
            raise RuntimeError("Train both prespecified models before final evaluation")
        if self.selection is not None:
            return self.selection
        # Python max retains the earlier entry on a tie: CNN wins ties.
        winner = max(("cnn", "resnet18"), key=lambda name: self.summaries[name]["best_val_macro_f1"])
        self.selection = {"criterion": "highest validation macro-F1; ties choose CNN", "winner": winner,
                          "models": self.summaries, "frozen_before_test": True,
                          "frozen_utc": datetime.now(timezone.utc).isoformat()}
        save_json(self.output / "selection.json", self.selection)
        return self.selection

    def evaluate_final(self):
        self._check_config()
        if self.selection is None:
            raise RuntimeError("Freeze model selection before loading test data")
        for name in ("cnn", "resnet18"):
            if sha256(self.output / name / "best.pt") != self.selection["models"][name]["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint changed after validation selection")
        # Re-running the cell only displays the saved comparison; it never refits.
        if (self.output / "comparison.csv").exists():
            return pd.read_csv(self.output / "comparison.csv")
        test = official_dataset("test", self.root)
        counts = pd.read_csv(self.output / "exploration/class_counts_before_test.csv", index_col=0)
        counts["official_test_final_only"] = np.bincount(test.labels.reshape(-1), minlength=9)
        counts.to_csv(self.output / "exploration/class_counts_all_splits.csv")
        manifest_path = self.output / "data_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(test_accessed_for_analysis=True, test_used=len(test))
        save_json(manifest_path, manifest)
        rows = []
        for name in ("cnn", "resnet18"):
            # Reload frozen weights without another pretrained download.
            model = build_model(name, pretrained=False).to(self.device)
            checkpoint = torch.load(self.output / name / "best.pt", map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
            loss, truth, probs = pass_batches(model, self._loader(name, test), self.device)
            metrics = write_evaluation(self.output / name / "final_test", truth, probs, CLASS_NAMES, test.imgs,
                                       f"PathMNIST final test — {name}; {self.config.profile}; train n={len(self.indices)}")
            rows.append({"model": name, "profile": self.config.profile, "train_n": len(self.indices),
                         "test_n": len(test), "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"],
                         "test_loss": loss, "selected_by_validation": name == self.selection["winner"],
                         **self.summaries[name]})
            del model
        comparison = pd.DataFrame(rows)
        comparison.to_csv(self.output / "comparison.csv", index=False)
        save_json(self.output / "status.json", {"status": "completed", "dataset": "PathMNIST",
                  "profile": self.config.profile, "test_used_for_selection": False,
                  "selected_model": self.selection["winner"], "test_samples": len(test)})
        lines = ["# Measured PathMNIST results", "", "Generated from saved predictions after validation-only checkpoint selection.", "",
                 "| Model | Train n | Test n | Accuracy | Macro-F1 | Best epoch |", "|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['model']} | {row['train_n']} | {row['test_n']} | {row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} | {row['best_epoch']} |")
        lines += ["", f"Validation-selected model: {self.selection['winner']}. Both prespecified models are reported; do not pick a winner using test scores.",
                  "Single-seed patch-level experiment. This is not clinical validation. Do not tune on these test errors."]
        (self.output / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        return comparison

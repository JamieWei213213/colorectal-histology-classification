# Colorectal Histology Classification with Deep Learning

An explainable portfolio experiment comparing a compact CNN trained from scratch with an ImageNet-pretrained ResNet18 on official nine-class PathMNIST.

**Status:** implementation complete; PathMNIST training and test evaluation have **not** run in this delivery environment. Local synthetic tests check software behavior only. See `EXECUTION_STATUS.md`. Run the notebook on a Colab GPU to produce biomedical results. This is an educational research project, not a clinically validated diagnostic tool.

## Run in Google Colab

1. Open https://colab.research.google.com/ and upload `Colorectal_Histology_Classification.ipynb`.
2. Select **Runtime > Change runtime type > GPU**. The planned batch size is intended for a single T4-class GPU; availability varies.
3. Run cells **1 through 8 in order**: install dependencies, write embedded code, set the protocol, download/explore data, train CNN, train ResNet18, freeze selection, evaluate test.
4. Cell 9 provides the error-review worksheet. Cell 10 downloads source, metrics, plots and checkpoints. Save the executed notebook separately from Colab's File menu.

The notebook is **self-contained**: no project ZIP, GitHub repository, manual data upload or API key is needed. Internet is required for packages, official data and pretrained weights. The default `PROFILE='today'` is a real training-subset experiment. Set `PROFILE='full'` **before the first run** to use all training images. Do not tune after viewing test results.

## Question and dataset

Under a modest fixed budget, how do these two practical pipelines differ in accuracy, macro-F1, class errors and cost? Architecture, preprocessing, input size and trainable parameters differ, so this is not a controlled estimate of pretraining alone.

Histology means examining thin tissue slices under a microscope. H&E uses hematoxylin to emphasize blue-purple nuclear material and eosin to emphasize pink/red cytoplasm and supporting tissue. A patch is a small crop from a larger slide. The model labels patches; it does not diagnose patients or estimate survival, immune function or treatment benefit.

| ID | Official class | Plain meaning |
|---|---|---|
| 0 | adipose | Fat tissue |
| 1 | background | Mostly non-tissue area |
| 2 | debris | Tissue/cellular remnants |
| 3 | lymphocytes | A category containing immune cells |
| 4 | mucus | Secreted gel-like material |
| 5 | smooth muscle | Involuntary muscle tissue |
| 6 | normal colon mucosa | Normal inner colon lining |
| 7 | cancer-associated stroma | Supporting tissue associated with a tumor |
| 8 | colorectal adenocarcinoma epithelium | Gland-forming colorectal cancer epithelial tissue |

Official train/validation/test sizes: **89,996 / 10,004 / 7,180**. Source images here are RGB 28x28. MedMNIST derives train/validation from NCT-CRC-HE-100K and test from CRC-VAL-HE-7K; original source patches are 224x224. The original source archive reports no patient overlap between its 100K and 7K cohorts. **This project has not audited patient IDs and does not claim patient-disjoint MedMNIST train/validation splits.** The six image/label arrays used here do not provide patient IDs; related patches can be correlated.

`PathMNIST(split='train'|'val'|'test', size=28, download=True)` downloads the official Zenodo archive with the package's checksum check. No combined-dataset random split is made. The archive physically contains all splits; test arrays/labels are not inspected by this workflow until checkpoint/model selection is frozen.

Dataset license: **CC BY 4.0**. Credit MedMNIST and original authors, link the license and identify preprocessing changes when reusing data/figures. Upstream MedMNIST code is Apache-2.0. See `docs/sources.md`. Data and pretrained weights are not bundled. Source images are resized/normalized as specified below.

## Methods and bounded budget

| Setting | Compact CNN | Pretrained ResNet18 |
|---|---|---|
| Source | Same 28x28 RGB images | Same 28x28 RGB images |
| Input | 28x28; mean/std 0.5 | Official IMAGENET1K_V1 preset: resize 256, crop 224, ImageNet normalization |
| Architecture | Three conv-BN-ReLU-pool blocks; global pool; dropout; nine-class head | Residual backbone; replace head with 512-to-9 linear layer |
| Trained layers | All | `layer4` and `fc` only |
| AdamW learning rate | 0.001 | 0.0003 |
| Today maximum epochs | 6 | 4 |
| Full maximum epochs | 8 | 6 |

Common settings: seed 42, batch 64, weight decay 0.0001, cosine LR schedule, cross-entropy, gradient clipping, CUDA mixed precision, early stopping after two non-improving validation epochs. Frozen ResNet batch-normalization statistics remain frozen. No class weights or test-derived normalization are fitted. Only training receives horizontal/vertical random flips; val/test transforms are deterministic.

Today mode selects **20,000 proportionally stratified training images**, saving their official indices. All 10,004 validation and 7,180 test images are used. Full mode trains on all 89,996 training images. Epoch budgets bound optimizer steps; no runtime guarantee is made. CPU fallback works but is slower. If GPU memory is insufficient, start a fresh run with a smaller batch before training.

Upsampling to 224 matches the pretrained pipeline but adds no biological detail. Its standard center crop also removes some border content. A future MedMNIST+ higher-resolution experiment needs a new protocol and is not equivalent to this one.

## Selection and outputs

Each model keeps the epoch with highest **validation macro-F1** (classes weighted equally; ties retain the earlier epoch). Overall model selection is also by validation only, with CNN winning ties. Both prespecified models receive final test evaluation for reporting; do not choose/tune using that table. Re-running the completed evaluation cell displays the saved comparison.

Saved outputs: accuracy; macro-F1; per-class precision, recall, F1 and support; raw/row-normalized confusion matrices; all predicted probabilities; first correct/incorrect examples in split order; curves; confusion pairs; measured comparison table; data/model/source hashes; checkpoints. Softmax probabilities are not calibrated confidence. `RESULTS.md` and completed status are written only after final evaluation.

**Real PathMNIST results: not yet run.** No invented values or borrowed benchmark results. `outputs/synthetic_smoke/` contains measured software-test outputs only. After Colab, use `outputs/<run>/RESULTS.md` and `comparison.csv` for biomedical results.

## Local setup and reproducibility

Use Python 3.10+ and a fresh virtual environment. Install a compatible torch/torchvision build for your platform from https://pytorch.org/get-started/locally/, then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py --profile today
# Select this BEFORE examining test results if a full run is intended:
python run_experiment.py --profile full
```

Colab's matched GPU torch/torchvision pair is retained. Dependencies are bounded rather than universally locked because CUDA builds differ by platform. Runs save resolved versions, `pip freeze`, source/data hashes, training indices and configuration. Seeds and deterministic settings improve repeatability; hardware/version differences and unsupported deterministic kernels remain possible. Checkpoints save model state/inference metadata, not optimizer/RNG state for exact training resumption. A disconnect needs a new run unless evaluation-only recovery is added.

## Structure and limitations

- `histology/core.py`: CNN, training, checkpointing, metrics and plots.
- `histology/data.py`: official splits, subset, transforms and ResNet18.
- `histology/experiment.py`: protocol, validation selection and final reporting.
- `run_experiment.py`: CLI; `build_notebook.py`: self-contained notebook builder.
- `tests/test_integrity.py`: known-answer metrics, subset checks and synthetic end-to-end training.
- `docs/`: interview guide, error review, sources and truthful resume bullets.

Limitations: low resolution; patch correlation; one seed; no patient-level inference; no prospective validation; no calibration/fairness, scanner or stain robustness studies; no pathologist-reviewed error explanations. A lymphocyte class does not measure immune activity. This project demonstrates reproducible biomedical ML experimentation, not clinical or wet-lab expertise.

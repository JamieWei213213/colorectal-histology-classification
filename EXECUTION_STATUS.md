# Actual execution status — 2026-10-20

## Completed locally

- Three unittest cases passed: known-answer metrics and invalid-probability rejection; deterministic proportional training-only subset selection; an end-to-end synthetic training/checkpoint/report test.
- CompactCNN trained for **4 epochs on 180 synthetic color patches**, with 90 separate synthetic validation patches and 90 synthetic evaluation patches.
- **94,633 trainable parameters**, measured from the model. Training cross-entropy fell from **1.09512** to **0.11428**. Validation macro-F1 tied across epochs, so the documented rule retained epoch 1.
- The selected checkpoint reloaded into a new model and produced **exactly identical prediction arrays**. Weight changes, finite loss and probability shapes were checked.
- Synthetic evaluation accuracy and macro-F1 were both **1.0000 on 90 trivial synthetic samples**. These are software-fixture measurements, **not PathMNIST or biological performance**, and must not be used as resume/benchmark results.
- Saved actual synthetic curves, checkpoint, metrics, predictions, confusion matrix and example figures under `outputs/synthetic_smoke/`. No incorrect examples occurred in this trivial fixture; that figure explicitly says so.
- All Python files and the notebook's ten code cells parsed successfully. Embedded notebook source matched the reusable modules exactly. The notebook contains no fabricated execution outputs.
- Learning-curve and confusion-matrix PNGs were visually checked for readable labels/layout.

## Not executed locally

Official PathMNIST download, real tissue exploration, ResNet18/pretrained-weight loading, torchvision transforms, GPU training, real validation/model comparison, and final official test evaluation. Consequently, there are **no measured PathMNIST scores or observed tissue-confusion conclusions** in this delivery.

The machine reported an RTX 4060, but this session's installed PyTorch was **2.11.0+cpu** and `torch.cuda.is_available()` was false. `torchvision` and `medmnist` were absent; shell network access failed DNS resolution. Packages could not be downloaded in this environment. An initial smoke attempt hit a sandbox temp-directory permission error; the test was corrected to use an isolated temporary directory under this project, then all three tests passed.

Local versions: Python 3.12.10; NumPy 2.3.1; pandas 2.3.0; scikit-learn 1.7.0; Matplotlib 3.10.8; Pillow 12.1.1. Machine-readable evidence is in `outputs/synthetic_smoke/execution.json`, `training_summary.json`, `history.csv` and `outputs/notebook_validation.json`.

## Exact remaining Colab steps

1. Upload `Colorectal_Histology_Classification.ipynb` at https://colab.research.google.com/.
2. Select a GPU runtime. Run Cells 1–4 for setup, embedded files, configuration, official download/exploration and both-model preflight.
3. Run Cells 5–6 to train the two models. Keep the predeclared profile and full validation split.
4. Run Cell 7 to freeze validation-only selection, then Cell 8 for the only final official test evaluation and measured reports.
5. Run Cell 9 for error review and Cell 10 to download results/checkpoints. Save the executed notebook separately.

Use the generated `RESULTS.md`/`comparison.csv` for numerical claims only after completion. If a download, dependency or GPU error occurs, preserve the traceback and the completed outputs; do not claim an incomplete run succeeded.

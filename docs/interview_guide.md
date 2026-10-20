# Interview guide

## Honest project status

The source implements the full PathMNIST experiment. The delivered local run checks the CNN pipeline on synthetic data; it is not a trained histology model. The real-data download, pretrained ResNet18, augmentation and GPU execution must run in Colab. Only describe achieved histology performance after reading that run's saved metrics. Anything below describing runtime behavior is implemented behavior, not a claim that the Colab experiment has already run.

## Architecture and data flow

`Config` defines the protocol before data analysis. `Experiment.prepare()` gets official train/validation data and records a seeded subset of training indices. `PatchDataset` converts array images to PIL RGB images, applies a model-specific transform, and returns an image tensor plus an integer label. `DataLoader` stacks examples into batches. `build_model()` constructs the CNN or pretrained ResNet18. `fit_model()` optimizes on training batches and selects checkpoints by validation macro-F1. `freeze_selection()` records the model choice and checkpoint hashes. `evaluate_final()` then loads the official test split, reloads those checkpoints, and generates reports from actual predictions.

There is no database, web service, patient record system or external LLM. Files are sufficient for a small reproducible experiment. Images stay in the downloaded NPZ archive; metrics and provenance use JSON/CSV, indices use NPY, weights use PyTorch PT files, plots use PNG.

## Important classes, functions and structures

| Component | Purpose and plain explanation |
|---|---|
| `Config` dataclass | A typed container for seed, budget, learning rates, batch size and profile |
| `Experiment` | Enforces stage order and connects data, models, training and reporting |
| `PatchDataset` | Adapts image arrays to the interface a PyTorch loader expects |
| `sample_training_indices` | Proportional class sampling within training only; saves exact membership |
| `CompactCNN` | Learns local image patterns with small convolution filters |
| `ResNet18` | A larger feature extractor with shortcut connections; starts with ImageNet weights |
| `pass_batches` | Runs one whole loader; updates gradients only if an optimizer is supplied |
| `fit_model` | Optimizes, monitors validation, early-stops and restores the best checkpoint |
| `score_predictions` | Validates probability shape and computes metrics from true/predicted classes |
| `write_evaluation` | Saves metrics, probabilities, confusion tables, examples and bounded error statements |
| `make_loader` | Batches samples and seeds shuffle/workers reproducibly |
| `seed_everything` | Seeds Python, NumPy and PyTorch and requests deterministic kernels |
| `model.state_dict()` | Mapping from layer names to learned weight/statistic tensors |

Dependencies: PyTorch provides tensors, automatic differentiation, optimizers and GPU execution; torchvision supplies ResNet18 and pretrained transforms; MedMNIST provides official data/splits; NumPy manages arrays; scikit-learn computes metrics and stratified sampling; pandas saves tables; Matplotlib draws plots; Pillow represents RGB images.

## Tensor shapes and one example

A raw patch is `uint8[28,28,3]`: height, width, red/green/blue channels, pixel values 0–255. Suppose the dataset label is 6 (normal colon mucosa). This is an illustrative path, not a measured prediction.

For the CNN, training may flip the patch. `ToTensor` produces `float32[3,28,28]` in [0,1]; fixed normalization maps it to roughly [-1,1]. A batch is `[B,3,28,28]`, with labels `int64[B]`. The three blocks yield `[B,32,14,14]`, `[B,64,7,7]`, then `[B,128,3,3]`. Global average pooling averages each feature map into `[B,128]`. The linear head produces `[B,9]` **logits**, unrestricted class scores.

Cross-entropy compares those logits with integer class 6. Automatic differentiation computes how each trainable weight affects loss; AdamW updates weights. Do not softmax before cross-entropy: PyTorch performs the stable equivalent internally. For reporting, softmax maps logits to nine values summing to one, and `argmax` returns the largest-score class. A label-6 patch predicted as 7 increments confusion-matrix row 6, column 7.

For ResNet18, the official preset enlarges to 256, crops 224, scales pixels and applies ImageNet channel means/stds. Inputs are `[B,3,224,224]`. Standard stages produce 64,128,256,512 feature channels; global pooling gives `[B,512]`; the new head gives `[B,9]`. The stem and first three residual stages are frozen; stage 4 and the head learn. Frozen batch-normalization running means/variances are kept unchanged.

## Design choices and tradeoffs

These are **documented implementation choices**, not inferred historical decisions:

- Official splits preserve benchmark membership. A newly randomized combined split could mix related patches and destroy test independence.
- A compact CNN gives an understandable from-scratch baseline. A linear pixel classifier would be cheaper but would encode spatial texture poorly; a very deep CNN costs more to train.
- ResNet18 is a practical transfer-learning comparison. A residual shortcut adds a block's input to its output, helping information and gradients travel through depth. Larger ResNets or a vision transformer add compute and need a separate comparison.
- Freeze earlier layers to limit training cost and allow later features to adapt. Full fine-tuning may adapt more but costs more and may overfit; freezing the entire backbone may adapt too little.
- Use official pretrained preprocessing to avoid an undocumented mismatch. Upsampling does not restore biological detail, and center cropping sacrifices borders. Higher-resolution MedMNIST+ or full-patch resizing are worthwhile future experiments.
- Macro-F1 gives each class equal importance even with unequal frequency. Accuracy alone can hide poor minority-class recall. Neither measures clinical usefulness.
- Use flips only in training as a simple orientation augmentation. This assumes patch label is unaffected by orientation; it is not an empirical proof that flips improve performance. Avoid arbitrary stain/color distortions without validation.
- Select with validation and report test once. Repeatedly changing the model after observing test errors turns the test set into a development set.
- Fixed epochs and a saved training subset serve today's compute budget. This is not an exhaustive search or a claim of optimal settings.

**Inferred/unverified hypotheses:** natural-image pretraining may supply useful color/texture features; low resolution may explain some confusions; similar visual patterns may affect specific pairs. These mechanisms have not been established by this project. Do not present them as observed biological findings or proven causes.

## 30-second explanation

“I built a reproducible PyTorch experiment for nine-category colorectal tissue-patch classification with official PathMNIST splits. It compares a compact CNN with pretrained ResNet18, selects checkpoints using validation macro-F1, and reserves the test set for final accuracy, per-class metrics and error analysis. I record the data subset, seeds, versions and checkpoints. The local software smoke test passed; the histology results will come from the GPU notebook. It is an educational benchmark, not a clinical diagnostic tool.”

After a successful Colab run, replace the execution-status sentence with measured results and the actual training size; do not insert numbers from a published benchmark or the synthetic test.

## Deeper explanation

“The task maps a small H&E image patch to one of nine tissue labels. I retain MedMNIST's train/validation/test assignment and sample only within training for the bounded default experiment. The CNN learns all layers at native 28-pixel resolution. ResNet18 uses explicit ImageNet preprocessing and adapts the last stage and classifier. A common training loop handles cross-entropy, AdamW, mixed precision, validation and checkpointing. I freeze both best checkpoints and the validation-selected model before loading the test split for analysis. Reports come directly from saved probabilities and include class-normalized confusion matrices and indexed examples. The comparison changes several pipeline factors, so I cannot attribute any difference solely to pretraining. I also cannot infer patient-level diagnostic performance, immune function or biological mechanisms from patch-level accuracy.”

## Eight likely questions and sample answers

1. **What exactly is the target?** One categorical tissue label per image patch. It is not a whole-slide or patient cancer diagnosis. A patch's tissue class and a patient's disease status are different units of analysis.
2. **How did you prevent leakage?** I kept official split membership, selected the subset only from training, used training-only random transforms and used validation macro-F1 for checkpoints/model choice. The test split is analyzed after those choices are frozen. I did not verify patient-disjoint train/validation membership.
3. **Why use a pretrained model on histology?** It is a prespecified practical comparison. Transfer of useful generic features is a plausible hypothesis; actual benefit must be measured. ImageNet differs substantially from histology, and this comparison is not a pure pretraining ablation.
4. **Why upscale 28 to 224?** To use the exact pretrained input pipeline. It adds computational cost but no missing microscopic detail. Training a higher-resolution source dataset would be a different experiment.
5. **Why macro-F1?** Precision asks how often predictions of a class are right; recall asks how many true members are found. F1 balances those two. Macro-F1 averages class F1 equally, making weak classes more visible than overall accuracy alone.
6. **What does freezing mean, and why freeze batch normalization?** Frozen parameters receive no gradient updates. Batch normalization also maintains running statistics that can change in training mode without gradients, so I explicitly keep frozen blocks in evaluation mode.
7. **How do you interpret a tissue confusion?** I report the measured direction, count and fraction of the true class, inspect indexed mistakes, and separate visible patterns from possible explanations. Downsampling or similar appearance is a hypothesis, not proof of tumor biology. A pathologist and higher-resolution review would improve interpretation.
8. **What would you improve next?** A prespecified higher-resolution and pretraining ablation, multiple seeds, patient/slide provenance audit, scanner/stain robustness and probability calibration. Those are future work. For clinical claims I would need much broader validation and expert collaboration.

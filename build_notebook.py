"""Build a standalone Colab notebook from the same tested Python modules."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent
cells = []


def add(kind, source, title=None):
    source = textwrap.dedent(source).strip() + '\n'
    cell = {'cell_type': kind, 'id': f'cell-{len(cells):02d}', 'metadata': {}, 'source': source.splitlines(True)}
    if kind == 'code':
        ast.parse(source)
        cell.update(execution_count=None, outputs=[])
        if title:
            cell['metadata']['cellView'] = 'form'
    cells.append(cell)


add('markdown', '''
# Colorectal Histology Classification with Deep Learning

**A reproducible, explainable portfolio experiment — official PathMNIST, nine tissue classes.**

Upload this notebook into Google Colab, select **Runtime → Change runtime type → GPU**, then run Cells **1–10 in order**. Everything needed is embedded below; no repository or project ZIP upload is required. The default uses 20,000 official training images, all validation images and all test images. Choose `full` in Cell 3 before starting if you want the entire training split.

**Execution disclosure:** no PathMNIST models were trained during notebook delivery. A separate local CPU synthetic software test verifies training, checkpoint reload and report generation; it does not measure histology performance. Real model tables and plots below appear only after you execute the cells. No results are prefilled. This is not a clinically validated diagnostic tool.

**Question:** under a bounded compute budget, how do a small CNN and pretrained ResNet18 differ in patch classification and error patterns? This compares full pipelines; it does not isolate the effect of pretraining.

## Biology in plain English
Histology studies thin tissue slices under a microscope. H&E combines hematoxylin, which emphasizes blue-purple nuclear material, and eosin, which emphasizes pink/red cell material and supporting tissue. A patch is a small image region, not a whole slide or a patient. Lymphocytes are immune cells; classifying a lymphocyte-rich patch does not measure immune function.

| ID | Official tissue class | Plain explanation |
|---|---|---|
| 0 | adipose | Fat |
| 1 | background | Non-tissue area |
| 2 | debris | Tissue/cellular remnants |
| 3 | lymphocytes | Immune cells |
| 4 | mucus | Secreted gel-like material |
| 5 | smooth muscle | Involuntary muscle |
| 6 | normal colon mucosa | Normal inner colon lining |
| 7 | cancer-associated stroma | Supporting tissue associated with tumor |
| 8 | colorectal adenocarcinoma epithelium | Gland-forming colorectal cancer epithelial tissue |

## Dataset provenance and boundaries
Official split sizes: **89,996 training / 10,004 validation / 7,180 test**. RGB images here are 28×28, derived from 224×224 source patches. MedMNIST uses NCT-CRC-HE-100K for train/validation and CRC-VAL-HE-7K for test. The original source archive reports separate patients between those source cohorts. We have **not independently audited patient IDs or verified patient-disjoint train/validation membership**. Official split membership is preserved; no reshuffling of combined splits occurs.

Data: CC BY 4.0. Attribute MedMNIST/Yang et al. and the original Kather et al. datasets, link the license and state modifications. MedMNIST code is Apache-2.0. The official archive is downloaded via `PathMNIST(..., download=True)`. Its bytes contain all splits, but test arrays are not analyzed until model selection is frozen. Sources and attribution appear at the end and in the embedded documentation.
''')
add('markdown', '''
## Cell 1 — Setup and hardware
Keep Colab's matched CUDA PyTorch/torchvision installation. Install missing project dependencies, then verify imports. CPU fallback is valid but slower. If you changed package versions in a previously used runtime and imports fail, start a fresh runtime rather than mixing torch/torchvision builds.
''')
add('code', '''
import sys, subprocess, importlib.metadata
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
    'medmnist==3.0.2', 'numpy>=1.26,<3', 'pandas>=2.1,<4',
    'scikit-learn>=1.4,<2', 'matplotlib>=3.8,<4', 'Pillow>=10,<13'])
import torch, torchvision, medmnist
print('Python:', sys.version)
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'MedMNIST:', medmnist.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE — CPU fallback; select a GPU runtime for practical training')
''')
add('markdown', '''
## Cell 2 — Materialize the reusable code
This cell writes the exact packaged modules and explanatory documents into `/content/colorectal_histology` (or the current directory outside Colab). It does not fetch a repository. You can inspect the `.py` files in Colab's Files panel. The rest of the notebook uses these modules directly.
''')
paths = sorted((ROOT/'histology').glob('*.py')) + sorted((ROOT/'docs').glob('*.md')) + [ROOT/'README.md', ROOT/'EXECUTION_STATUS.md', ROOT/'requirements.txt', ROOT/'run_experiment.py', ROOT/'tests/test_integrity.py']
bundle = {p.relative_to(ROOT).as_posix(): p.read_text(encoding='utf-8-sig') for p in paths}
bootstrap = '''
from pathlib import Path
import os, sys
PROJECT = Path('/content/colorectal_histology') if Path('/content').exists() else Path.cwd()/'colorectal_histology_colab'
PROJECT.mkdir(parents=True, exist_ok=True)
FILES = BUNDLE_LITERAL
for relative, content in FILES.items():
    destination = PROJECT/relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding='utf-8')
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
print('Project ready:', PROJECT, 'files:', len(FILES))
'''.replace('BUNDLE_LITERAL', repr(bundle))
add('code', bootstrap)
add('markdown', '''
## Cell 3 — Fix the experiment before viewing results

`today`: 20,000 proportionally stratified images from the official training split, max 6 CNN and 4 ResNet epochs. `full`: all 89,996 training images, max 8 and 6 epochs. Both use the same training membership for the two models and the complete validation/test splits.

Batch 64; early stopping patience 2; seed 42. Only training has random horizontal/vertical flips. An epoch budget bounds the number of optimizer steps; actual runtime depends on your GPU. Checkpoints, versions and training indices are saved. A fresh run directory prevents mixing results from different protocols. Change batch size here before starting if needed for GPU memory. Seeds improve repeatability but do not guarantee identical results across library/hardware versions.

Optional persistence: set `USE_DRIVE=True` to mount Google Drive and save run outputs there. This triggers Google's normal authorization in Colab. Otherwise outputs live in temporary Colab storage: download Cell 10 before disconnecting.
''')
add('code', '''
from datetime import datetime, timezone
from histology.experiment import Config, Experiment
from IPython.display import display, Image, Markdown
PROFILE = 'today'  # choose 'full' here BEFORE training or test inspection
USE_DRIVE = False
config = Config.for_profile(PROFILE)
# config.batch_size = 32  # optional memory adjustment, before creating the experiment
run_id = PROFILE + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    output_root = Path('/content/drive/MyDrive/colorectal_histology_runs')
else:
    output_root = PROJECT/'outputs'
RUN_DIR = output_root/run_id
experiment = Experiment(config, root=PROJECT/'data', output=RUN_DIR)
print(config)
print('Outputs:', RUN_DIR)
''')
add('markdown', '''
## Cell 4 — Official download, exploration and preflight
Download only the 28-pixel PathMNIST archive via the official package. Count/plot training and validation classes, show training examples, and save exact selected training indices. Test class counts are deferred to final evaluation. A two-image forward pass checks each model and downloads official pretrained weights; this is a preflight, not a performance experiment.
''')
add('code', '''
display(experiment.prepare())
display(Image(filename=str(RUN_DIR/'exploration/labeled_training_examples.png')))
display(Image(filename=str(RUN_DIR/'exploration/class_counts.png')))
print(experiment.preflight())
''')
add('markdown', '''
## Cell 5 — Compact CNN baseline
Three convolution blocks learn local patterns: 3→32→64→128 channels, each with batch normalization, ReLU and max pooling. Global average pooling turns spatial maps into 128 features; dropout and a linear layer produce nine logits. The native 28×28 input is scaled to [0,1], then normalized with fixed mean/std 0.5. Cross-entropy consumes raw logits and integer labels.

AdamW learning rate 0.001, cosine schedule, weight decay 0.0001. Validation macro-F1 chooses the checkpoint; test images never enter training. Early stopping follows two non-improving validation epochs. CUDA uses mixed precision and gradient scaling.
''')
add('code', '''
cnn_summary = experiment.train_one('cnn')
print(cnn_summary)
display(Image(filename=str(RUN_DIR/'cnn/learning_curves.png')))
''')
add('markdown', '''
## Cell 6 — Pretrained ResNet18 comparison
ImageNet pretraining supplies initial weights; its benefit here must be measured. ResNet shortcut connections add a block's input to its output. Replace the 512-to-1000 head with a 512-to-9 layer and train only the final residual stage (`layer4`) plus that head. Earlier layers and their batch-normalization running statistics stay frozen. AdamW LR is 0.0003.

We call `ResNet18_Weights.IMAGENET1K_V1.transforms()` exactly: bilinear resize to 256, center crop 224, scale to [0,1], normalize using ImageNet mean [0.485,0.456,0.406] and std [0.229,0.224,0.225]. Only training gets random flips beforehand. **Upsampling 28→224 does not restore microscopic detail**, and cropping discards some borders. Keeping the official preset makes the assumption explicit. A future MedMNIST+ experiment can test genuine higher-resolution input.
''')
add('code', '''
resnet_summary = experiment.train_one('resnet18')
print(resnet_summary)
display(Image(filename=str(RUN_DIR/'resnet18/learning_curves.png')))
''')
add('markdown', '''
## Cell 7 — Freeze selection before testing
Each model's best checkpoint uses validation macro-F1 (ties keep earlier epoch). Choose the overall model using validation macro-F1 only (model ties choose CNN). Save those choices and checkpoint hashes. Both fixed models will be reported on test, but do not switch the selection based on test performance.
''')
add('code', '''
selection = experiment.freeze_selection()
print(selection)
''')
add('markdown', '''
## Cell 8 — Final official test evaluation and actual comparison
This is the first stage that analyzes official test images/labels. Accuracy is the fraction correct. Precision asks how often predictions of a class are correct; recall asks how much of the true class was found; F1 balances both. Macro-F1 averages nine class F1 scores equally. Undefined precision/recall is reported as zero, and support shows the denominator.

The confusion matrix uses true classes as rows and predictions as columns. Class-normalized rows help compare different class sizes. Correct/incorrect grids show the first examples in split order, with indices traceable to the saved probabilities. Maximum softmax values are not calibrated clinical confidence. Tables are generated only from this run's measured predictions.
''')
add('code', '''
import pandas as pd
comparison = experiment.evaluate_final()
display(comparison[['model','train_n','test_n','best_epoch','best_val_macro_f1',
                    'test_accuracy','test_macro_f1','training_seconds','selected_by_validation']])
display(pd.read_csv(RUN_DIR/'exploration/class_counts_all_splits.csv'))
for name in ['cnn','resnet18']:
    report_dir = RUN_DIR/name/'final_test'
    display(Markdown('### ' + name))
    display(pd.read_csv(report_dir/'classification_report.csv', index_col=0))
    display(Image(filename=str(report_dir/'confusion_matrix.png')))
    display(Image(filename=str(report_dir/'correct_examples.png')))
    display(Image(filename=str(report_dir/'incorrect_examples.png')))
    display(Markdown((report_dir/'error_analysis.md').read_text()))
display(Markdown((RUN_DIR/'RESULTS.md').read_text()))
''')
add('markdown', '''
## Cell 9 — Error review, interview preparation and limits
The generated error report states measured confusions. Biological causes require expert review and additional evidence. Open the worksheet below, then inspect the top directed pairs and indexed examples. Distinguish “the model confused A with B” from an unverified explanation for why. Do not tune this experiment after seeing test errors.

Limitations: reduced resolution, correlated patches, one seed, unverified patient-level train/validation separation, no clinical validation, no calibration or scanner/stain robustness study. This does not measure prognosis, immune activity or treatment response. The embedded interview guide explains implementation, tensors, a sample data flow, design alternatives and eight follow-up questions. Its execution-status sentence must be updated truthfully after your completed run; resume bullets intentionally omit unmeasured performance.
''')
add('code', '''
display(Markdown((PROJECT/'docs/error_analysis.md').read_text()))
for name in ['cnn','resnet18']:
    print(name, 'largest measured directed confusions:')
    display(pd.read_csv(RUN_DIR/name/'final_test/confusion_pairs.csv').head(5))
print('Interview guide:', PROJECT/'docs/interview_guide.md')
print('Resume bullets:', PROJECT/'docs/resume_bullets.md')
''')
add('markdown', '''
## Cell 10 — Save the completed experiment
Downloads code, documentation, plots, predictions, metrics and best checkpoints. Raw data and cached ImageNet weights are excluded because the notebook downloads official copies. Download the executed notebook separately so your visible outputs are retained. Checkpoints support inference; exact interrupted-training resume is not implemented.
''')
add('code', '''
import zipfile
archive = PROJECT/(run_id + '_results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for folder in ['histology','docs','tests']:
        for path in (PROJECT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                z.write(path, path.relative_to(PROJECT).as_posix())
    for name in ['README.md','EXECUTION_STATUS.md','requirements.txt','run_experiment.py']:
        z.write(PROJECT/name, name)
    for path in RUN_DIR.rglob('*'):
        if path.is_file():
            z.write(path, 'outputs/'+run_id+'/'+path.relative_to(RUN_DIR).as_posix())
print('Saved:', archive, 'bytes:', archive.stat().st_size)
try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    print('Outside Colab: open the ZIP at the path printed above.')
''')
add('markdown', (ROOT/'docs/sources.md').read_text(encoding='utf-8-sig'))
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
             'language_info': {'name':'python','version':'3.12'}, 'colab': {'name':'Colorectal_Histology_Classification.ipynb'},
             'accelerator':'GPU'}, 'nbformat':4, 'nbformat_minor':5}
target = ROOT/'Colorectal_Histology_Classification.ipynb'
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
code_cells = [c for c in cells if c['cell_type'] == 'code']
assert len(code_cells) == 10
assert all(c['execution_count'] is None and c['outputs'] == [] for c in code_cells)
# Validate embedding byte-for-byte so the standalone notebook uses the reviewed code.
parsed = ast.parse(''.join(code_cells[1]['source']))
embedded = next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FILES' for t in n.targets))
assert embedded == bundle
for path in ROOT.rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
validation = {'notebook_format':'4.5', 'code_cells':len(code_cells), 'total_cells':len(cells),
              'all_code_cells_parse':True, 'embedded_files_match_source':True, 'embedded_files':len(bundle),
              'no_fabricated_execution_outputs':True,
              'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'validation':'JSON structure, Python AST and embedding checks; official-data notebook execution remains for Colab'}
(ROOT/'outputs/notebook_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
print(json.dumps(validation, indent=2))

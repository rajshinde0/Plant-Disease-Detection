# Plant Disease Detection using ResNet9

Rebuild of the ASEP project described in `final_research_paper_ASEP_new.docx`:
a ResNet9 convolutional network that classifies a crop leaf photo into one of
**38 plant-disease classes across 14 crops**, and returns a treatment plan from
a local remedy database. A Flask web app wraps the model.

The whole system runs **offline**. That is a requirement, not an optimisation:
the paper's argument is that farmers often work where connectivity is worst, so
a remedy that needs the internet is unavailable exactly when it is needed.

> **Note on the two older files in this folder.** `random_forest.py` and
> `Random_Forest.ipynb` are **not** part of this project - they are a
> malicious-URL classifier (scikit-learn RandomForest over 79 URL features).
> They are left untouched. No plant-disease code survived, so this project was
> rebuilt from the research paper and the March 2025 presentation, which are
> the authoritative specification.

---

## Quick start

```bash
# 1. Extract the dataset from the Kaggle zip (~3 minutes)
python scripts/prepare_dataset.py

# 2. Check your machine and get a recommended training recipe
python scripts/check_env.py

# 3. Train (pick a recipe from the table below)
python -m src.train --img-size 64 --subset 0.3 --epochs 4

# 4. Measure real accuracy and regenerate every figure
python -m src.evaluate

# 5. Classify an image from the command line
python -m src.predict --image "data/raw/test/AppleCedarRust1.JPG"

# 6. Run the web app  ->  http://127.0.0.1:5000
python app/app.py
```

> **Windows path warning.** This folder's name contains `&` and `()`. Always
> quote paths in shell commands, or the shell will split on the `&`:
> `cd "C:/Users/.../Applied Science & Engineering Projects(ASEP)"`

---

## The dataset

[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- 87,900 RGB images, 38 classes, 80/20 split, plus 33 held-out test images.

| Split | Images | Classes |
|---|---|---|
| train | 70,295 | 38 |
| valid | 17,572 | 38 |
| test  | 33 | unlabelled |

Download the zip from Kaggle (~2.7 GB), then run `scripts/prepare_dataset.py`.
It defaults to `C:/Users/User/Downloads/archive (1).zip`; use `--zip` otherwise.

### Why you should not unzip it in File Explorer

The archive holds **175,767 entries - the entire dataset twice**:

| Entry | Files | Size |
|---|---|---|
| `New Plant Diseases Dataset(Augmented)/` | 87,867 | 1.43 GB |
| `new plant diseases dataset(augmented)/` | 87,867 | 1.43 GB (duplicate) |
| `test/` | 33 | - |

The two folder names differ only in case. Linux treats them as separate
directories; **Windows is case-insensitive**, so Explorer writes the second copy
over the first - 87,867 overwrite collisions, one per file - on top of its
single-threaded zip handler and Defender scanning all 175k files. Extraction
takes hours.

`prepare_dataset.py` extracts only the canonical subtree plus the test images,
flattens the doubled nesting, and uses a thread pool. **Measured: 2.9 minutes
at 505 files/s.** It is resumable, so an interrupted run continues where it
stopped. For an extra speedup, add the Defender exclusion it prints before
running.

```bash
python scripts/prepare_dataset.py                    # everything (1.43 GB)
python scripts/prepare_dataset.py --per-class 300    # quick subset
python scripts/prepare_dataset.py --zip "D:/x.zip" --workers 12
```

---

## Training

ResNet9 converges on this dataset by **epoch 2-3**, so the paper's stated 20
epochs is unnecessary - use `--epochs 20` only to reproduce its exact protocol.

### On a CPU laptop

Measured on an Intel i7-1065G7 (4 cores / 8 threads, 15 W):

| Image size | Throughput | Per epoch (70,295 images) |
|---|---|---|
| 64px | 40.8 img/s | 29 min |
| 128px | 10.7 img/s | 1.8 h |
| 256px | 2.8 img/s | 6.9 h |

Pick a recipe accordingly:

| Recipe | Command | Time | Expected |
|---|---|---|---|
| Smoke test | `--img-size 64 --subset 0.1 --epochs 1` | ~15 min | proves the loop |
| Quick | `--img-size 64 --subset 0.3 --epochs 4` | ~40 min | ~93-96% |
| Fast full | `--img-size 64 --epochs 4` | ~2 h | ~95-97% |
| Balanced | `--img-size 128 --subset 0.4 --epochs 3` | ~2.5 h | ~97-98.5% |
| Best on CPU | `--img-size 128 --epochs 3` | ~5.5 h | ~98-99% |

Avoid 256px on a CPU - roughly 7 hours per epoch.

Subsetting costs less accuracy here than it normally would, because this
dataset is **already offline-augmented**: its 87,867 images derive from ~54k
originals via rotations and flips, so a stratified subset still covers nearly
every distinct source leaf.

### Running a long job in the background

A multi-hour CPU run should not depend on a terminal staying open. Launch it
detached (PowerShell), with `-u` so the log updates live:

```powershell
$env:PYTHONIOENCODING = "utf-8"
Start-Process -FilePath "C:\Python314\python.exe" `
  -ArgumentList "-u","-m","src.train","--img-size","128","--epochs","3","--batch-size","32" `
  -WorkingDirectory "$PWD" `
  -RedirectStandardOutput "training.log" -RedirectStandardError "training.err.log" `
  -WindowStyle Hidden
```

Check on it at any time:

```powershell
Get-Content training.log -Tail 20          # per-epoch results
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*src.train*' }   # still running?
```

The checkpoint and `models/history.json` are rewritten after **every** epoch, so
an interrupted run still leaves the best model so far plus real training curves.

### On a machine with an NVIDIA GPU

This is the fast path to the paper's number - roughly 30 minutes for ~99%.

```bash
# 1. Install the CUDA build of PyTorch FIRST. Plain `pip install torch` gives
#    the CPU wheel, which ignores the GPU silently - no error, just 20x slower.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Confirm the GPU is visible (must print "CUDA available: True")
python scripts/check_env.py

# 3. Train at the paper's resolution with mixed precision
python -m src.train --img-size 256 --epochs 5 --batch-size 64 --amp
```

Drop `--batch-size` to 32 on a 4 GB card. `check_env.py` recommends a batch
size from your actual VRAM.

**Moving the project to another machine:** copy the folder (it is tiny - the
dataset and weights are excluded), copy the dataset zip or re-download it, then
run steps 1-3 above. No code changes are needed; the code picks the device
itself. To bring a model back, copy `models/plant_disease_resnet9.pth` and
`models/class_names.json` (~25 MB total).

### Integrated graphics cannot help

Intel Iris/UHD graphics cannot train this model.
`intel-extension-for-pytorch`'s XPU backend targets Arc, Core Ultra and Data
Center GPUs - not Gen11 Iris Plus. `torch-directml` supports DirectX 12
integrated GPUs but pins to much older torch and Python <= 3.12, so it will not
install against current versions, and on integrated graphics sharing system RAM
it is typically no faster than the CPU anyway. OpenVINO is inference-only.

---

## Evaluation and report figures

```bash
python -m src.evaluate              # metrics + confusion matrix + curves
python scripts/make_sample_grid.py  # dataset sample grid + class distribution
python -m src.compare_models        # ResNet6 vs ResNet9 table and chart
python -m src.lr_sweep              # learning-rate sweep
```

Outputs land in `reports/`:

| File | Contents |
|---|---|
| `metrics.json` | accuracy, top-3, macro/weighted precision/recall/F1, per-class table, confusion matrix, most-confused pairs |
| `classification_report.txt` | scikit-learn per-class report |
| `figures/confusion_matrix.png` | 38x38, row-normalised |
| `figures/per_class_f1.png` | per-class F1, worst classes labelled |
| `figures/accuracy_vs_epochs.png`, `figures/loss_vs_epochs.png` | training curves |
| `figures/dataset_samples.png`, `figures/class_distribution.png` | dataset figures |
| `figures/model_comparison.png`, `figures/lr_sweep.png` | paper comparisons |

### Corrections to the original paper

Everything above is **measured**, never copied from the write-up.

1. **The averaging method is unstated.** The paper reports precision 0.94,
   recall 0.96, F1 0.92. That looks self-contradictory but is not: macro-F1 is
   the *mean of per-class F1 scores*, not the harmonic mean of macro-precision
   and macro-recall, so it can legitimately fall below both. A class with
   precision 1.00 and recall 0.02 scores F1 0.04 and drags the average down -
   this project's own evaluation output shows exactly that pattern. What the
   paper omits is which averaging it used, so `evaluate.py` reports macro and
   weighted side by side.
2. **The optimal learning rate is disputed.** The paper says 0.1; the canonical
   ResNet9 implementation this architecture derives from uses 0.01.
   `src/lr_sweep.py` settles it empirically and prints whether the paper's
   claim reproduces.

The paper's error analysis (confusion between visually similar mildews, low
light, occluded leaves) is tested directly: `evaluate.py` reports
mildew-vs-mildew confusions specifically, and the app warns on low confidence.

---

## Project structure

```
config.py                  paths, dataset facts, training defaults
data/
  raw/                     extracted dataset (gitignored)
  remedies.json            38 classes: symptoms, organic, chemical, prevention
scripts/
  prepare_dataset.py       threaded extraction + verification
  check_env.py             device report + recommended settings
  make_sample_grid.py      dataset figures
src/
  model.py                 ResNet9 and the ResNet6 baseline
  data.py                  transforms, loaders, device helpers
  engine.py                One-Cycle training loop with gradient clipping + AMP
  train.py                 training CLI
  evaluate.py              metrics, confusion matrix, figures
  inference.py             shared model loading + single-image prediction
  predict.py               inference CLI
  remedies.py              offline remedy lookup
  compare_models.py        ResNet6 vs ResNet9
  lr_sweep.py              learning-rate sweep
  viz.py                   shared figure styling and validated palette
app/
  app.py                   Flask: /, /predict, /api/predict, /health
  templates/, static/
models/                    trained weights (gitignored)
reports/                   metrics and figures
tests/test_smoke.py        run with: python -m pytest tests/ -v
```

### Architecture

```
conv1: 3   -> 64
conv2: 64  -> 128, MaxPool(4)
res1:  [128->128, 128->128]  + skip
conv3: 128 -> 256, MaxPool(4)
conv4: 256 -> 512, MaxPool(4)
res2:  [512->512, 512->512]  + skip
classifier: AdaptiveMaxPool2d(1) -> Flatten -> Linear(512, 38)
```

6.59M parameters. Adam, One-Cycle LR, gradient clipping (0.1), weight decay
(1e-4), cross-entropy - all as the paper specifies.

One deliberate deviation: the classifier ends in `AdaptiveMaxPool2d(1)` rather
than a fixed `MaxPool2d(4)`. At 256px the two are numerically identical, but the
fixed version **crashes at 64px** because the feature map reaches 1x1 before the
final pool. The adaptive version lets the same weights-compatible model train at
64, 128 or 256px, which is what makes the CPU path possible.

Preprocessing is defined once in `src/data.py` and imported by training,
evaluation, the CLI and the web app, so inference can never drift away from how
the model was trained.

---

## Web app

```bash
python app/app.py     # http://127.0.0.1:5000
```

- Drag-and-drop or file-picker upload, with type and size validation
- Healthy / diseased verdict with a confidence meter and the top-3 shortlist
- Symptoms, organic treatment, chemical treatment and prevention
- Bundled sample images, so it demos without a photo to hand
- A low-confidence advisory below 60%, tied to the paper's error analysis
- `POST /api/predict` returns JSON; `GET /health` reports model state
- Starts and explains itself even with no trained model present

```bash
curl -F "image=@data/raw/test/AppleCedarRust1.JPG" \
     http://127.0.0.1:5000/api/predict
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Extraction takes hours | You are using Explorer. Use `scripts/prepare_dataset.py`. |
| `bash: Engineering: command not found` | Unquoted path - the `&` splits the command. Quote it. |
| Training is 20x slower than expected on a GPU box | You have the CPU wheel. Run `check_env.py`; reinstall from the cu124 index. |
| `Expected more than 1 value per channel` | A final training batch of 1 sample. Handled by `should_drop_last` in `src/data.py`. |
| `CUDA out of memory` | Halve `--batch-size`, or lower `--img-size`. |
| App says "No trained model yet" | Expected before training. Run `src.train`. |
| Predictions look random | The checkpoint's `img_size` must match training; it is stored in the checkpoint and applied automatically. Retrain if you edited transforms. |

---

## Credits

Applied Science & Engineering Project (ASEP), Computer Science Engineering,
Vishwakarma Institute of Technology, Pune. Project guide: Prof. Vijay U. Rathod.

Remedy guidance is compiled from standard agricultural extension sources and is
general advice only - follow local label rates and consult an extension officer
before applying any chemical treatment.

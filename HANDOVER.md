# HANDOVER — Plant Disease Detection (ResNet9)

Everything needed to move this project to GitHub, run it on the GPU laptop, retrain
it, and deploy it. Written for the migration to the NVIDIA gaming laptop.

- **Status:** complete and working. Trained model at **98.36%** validation accuracy.
- **Last trained:** 17 Oct 2026, 128px, 3 epochs, 418 min, Intel i7-1065G7 (CPU only).
- **Read `README.md` for full detail.** This file is the short operational runbook.

---

## 1. Current results (measured, not copied from the paper)

Model: ResNet9, 6.59M parameters, trained at 128x128 on all 70,295 training images.
Evaluated on the full 17,572-image validation set.

| Metric | Macro | Weighted |
|---|---|---|
| **Accuracy** | **98.36%** | — |
| **Top-3 accuracy** | **99.93%** | — |
| Precision | 0.9837 | 0.9838 |
| Recall | 0.9836 | 0.9836 |
| F1 | 0.9835 | 0.9836 |
| Cross-entropy loss | 0.0506 | — |

Held-out test images: **30/33** crop match, mean confidence **95.61%**.
Inference: **44.7 ms/image** at 128px on CPU.

### Training curve

| Epoch | val_acc | train_loss | val_loss | Time |
|---|---|---|---|---|
| 1 | 57.58% | 0.9974 | 1.6098 | 155.9 min |
| 2 | 87.73% | 0.3753 | 0.3876 | 130.0 min |
| 3 | **98.36%** | 0.1225 | 0.0505 | 132.2 min |

Total 418.2 min (6.97 h). Hyperparameters: Adam, One-Cycle LR (max_lr 0.01),
weight decay 1e-4, gradient clipping 0.1, cross-entropy, batch 32, seed 42.

Two things to understand before citing these:

1. **val_loss (0.051) is lower than train_loss (0.123).** This looks backwards but
   is correct: training sees augmented images (rotation, flips, colour jitter,
   random crops) which are harder than the clean validation images. It is evidence
   the augmentation is working, not a bug.
2. **Macro and weighted metrics agree to within 0.0002.** That means accuracy is
   spread evenly across all 38 classes — no class is quietly failing.

Best class: Corn (Maize) Healthy, F1 0.9989. Worst: Tomato Target Spot, F1 0.9320.

### Where the model actually errs

| Confusion | Rate | Why it is a credible error |
|---|---|---|
| Tomato Late Blight -> **Potato** Late Blight | 5.4% | *Same pathogen* (Phytophthora infestans) on two related crops |
| Tomato Target Spot -> Two-Spotted Spider Mite | 5.0% | Both produce fine leaf stippling |
| Corn NLB <-> Cercospora / Gray Leaf Spot | ~3% | Both are elongated grey-tan lesions along the veins |

**This corrects the research paper.** The paper claims confusion between mildew and
downy mildew. `src/evaluate.py` tests that claim explicitly and reports
**"none observed"**. The real top error is the same disease organism across two
host plants, which is a stronger finding — and the remedy database already handles
it correctly, since both entries prescribe the same treatment.

---

## 2. Putting this on GitHub

`git init` has **not** been run. From the project folder:

```bash
git init
git add .
git commit -m "Plant disease detection using ResNet9 - 98.36% validation accuracy"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

`.gitignore` is already written and correctly excludes the big and regenerable things:

| Excluded | Size | Why |
|---|---|---|
| `*.docx`, `*.pptx`, `*.pdf` | 2.6 MB | **Academic documents — deliberately not published** |
| `data/raw/` | **1.6 GB** | Re-extract from the Kaggle zip; never commit this |
| `models/*.pth` | 26 MB | Regenerate by training |
| `reports/figures/*.png`, `metrics.json` | 5.5 MB | Regenerate with `src.evaluate` |
| `training.log`, `webapp.log` | — | Run logs |
| `app/static/uploads/*` | — | User uploads |

The research paper, all three `.pptx` decks and the plagiarism report are excluded
by extension patterns rather than by filename. That is deliberate: gitignore treats
`[1]` as a character class, so a literal `ASEP_Endsem1_march[1][1].pptx` entry would
*not* match the actual file without escaping. The extension rules are also
future-proof — any further decks or drafts you drop in this folder stay private
automatically. Verified against the real filenames with `git check-ignore`.

What actually gets committed is **~400 KB of code**. That is the point: the repo
stays small and everything heavy is reproducible.

### Decisions to make before pushing

- **The trained model is gitignored.** If you want the 98.36% weights on the gaming
  laptop without retraining, either commit it deliberately
  (`git add -f models/plant_disease_resnet9.pth`, 26 MB — fine for GitHub, whose
  limit is 100 MB) or just copy the file across by USB. You said you plan to
  retrain, so this is optional insurance.
- **Academic documents are already excluded.** The research paper, the three decks
  and the plagiarism report will not be pushed. Confirmed by test.
- `random_forest.py` and `Random_Forest.ipynb` belong to a **different project**
  (a malicious-URL classifier) and are *currently still included*. They are
  unrelated to plant disease detection — add them to `.gitignore` if you would
  rather they stayed out.
- **Verify before your first push** — this prints exactly what would be committed:

  ```bash
  git add -A --dry-run
  git status --ignored --short       # lines starting with !! are excluded
  ```

---

## 3. Setting up on the GPU gaming laptop

### Step 1 — Install the CUDA build of PyTorch (do this first)

This is the single most important step. Plain `pip install torch` gives the
**CPU-only wheel**, which ignores the GPU silently — no error message, just 20x
slower training.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check your driver's CUDA version with `nvidia-smi` first and match the index URL
(`cu121`, `cu124`, `cu126`...).

### Step 2 — Verify the GPU is actually visible

```bash
python scripts/check_env.py
```

**Must print `CUDA available: True`** plus your GPU name and VRAM. If it says the
CPU-only wheel is installed, go back to step 1 — do not start training, because it
will take a day instead of half an hour. The script also recommends a batch size
based on your actual VRAM.

### Step 3 — Get the dataset

Download the zip from Kaggle
(<https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset>, ~2.7 GB)
or copy `archive (1).zip` across by USB, then:

```bash
python scripts/prepare_dataset.py --zip "path/to/archive.zip"
```

**Do not unzip it in File Explorer.** The archive contains 175,767 entries — the
entire dataset *twice*, in two folders whose names differ only in case
(`New Plant Diseases Dataset(Augmented)/` and `new plant diseases dataset(augmented)/`).
Windows is case-insensitive, so Explorer writes the second copy over the first:
87,867 overwrite collisions, one per file. That is what turns a few minutes into
hours. The script extracts only the canonical subtree — **measured 2.9 minutes at
505 files/s** — and verifies 70,295 / 17,572 / 33 before it finishes. It is
resumable, so an interrupted run continues where it stopped.

---

## 4. Retraining on the GPU

### The command

```bash
python -m src.train --img-size 256 --epochs 5 --batch-size 64 --amp
```

Roughly **30 minutes** for ~99%. This is the paper's resolution (256px) with mixed
precision, which is CUDA-only and roughly halves epoch time.

| Flag | Meaning |
|---|---|
| `--img-size 256` | the paper's resolution; only practical on a GPU |
| `--epochs 5` | ResNet9 converges by epoch 2-3; use `--epochs 20` only to reproduce the paper's exact stated protocol |
| `--batch-size 64` | drop to 32 on a 4 GB card; `check_env.py` recommends a value |
| `--amp` | mixed precision — CUDA only, silently ignored on CPU |

Other useful flags: `--subset 0.3` (fraction of each class, stratified),
`--no-augment`, `--max-lr`, `--num-workers`, `--model resnet6`, `--out <path>`.

**If you hit `CUDA out of memory`:** halve `--batch-size`, then lower `--img-size`.

### Running it detached (so it survives closing the terminal)

```powershell
$env:PYTHONIOENCODING = "utf-8"
Start-Process -FilePath "python" `
  -ArgumentList "-u","-m","src.train","--img-size","256","--epochs","5","--batch-size","64","--amp" `
  -WorkingDirectory "$PWD" `
  -RedirectStandardOutput "training.log" -RedirectStandardError "training.err.log" `
  -WindowStyle Hidden
```

Monitor it:

```powershell
Get-Content training.log -Tail 20
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*src.train*' }    # still running?
```

The `-u` flag matters — without it Python buffers stdout and the log stays empty.

**Interruption is safe.** The checkpoint and `models/history.json` are rewritten
after *every* epoch, so a stopped run still leaves the best model so far plus real
training curves. (This was added after a run was killed at epoch 2 and lost its
history.)

### What training writes

| File | Contents |
|---|---|
| `models/plant_disease_resnet9.pth` | best-epoch weights + class names + img_size + history |
| `models/class_names.json` | the 38 class names, in model output order |
| `models/history.json` | per-epoch loss/accuracy/LR, rewritten each epoch |

The checkpoint stores the `img_size` it was trained at, so inference configures
itself automatically. You never pass `--img-size` to the app or to `predict`.

---

## 5. Evaluation and report figures

```bash
python -m src.evaluate                 # full validation set (~13 min on CPU, far less on GPU)
python scripts/make_sample_grid.py     # dataset sample grid + class distribution
python -m src.compare_models           # ResNet6 vs ResNet9 (not yet run)
python -m src.lr_sweep                 # tests the paper's "lr 0.1 is optimal" claim (not yet run)
```

Useful flags: `src.evaluate --limit 0.1` evaluates on 10% for a quick check;
`--no-figures` skips plotting.

Outputs:

| File | Contents |
|---|---|
| `reports/metrics.json` | accuracy, top-3, macro+weighted P/R/F1, per-class table, full confusion matrix, most-confused pairs, history |
| `reports/classification_report.txt` | scikit-learn per-class report |
| `reports/figures/confusion_matrix.png` | 38x38, row-normalised |
| `reports/figures/per_class_f1.png` | per-class F1, weakest labelled |
| `reports/figures/accuracy_vs_epochs.png` | training curve |
| `reports/figures/loss_vs_epochs.png` | train vs validation loss |
| `reports/figures/dataset_samples.png` | one labelled leaf per class |
| `reports/figures/class_distribution.png` | images per class, train vs validation |
| `reports/figures/model_comparison.png` | ResNet6 vs ResNet9 (after `compare_models`) |
| `reports/figures/lr_sweep.png` | accuracy vs learning rate (after `lr_sweep`) |

---

## 6. Deployment — the web app

```bash
python app/app.py          # http://127.0.0.1:5000
```

Runs **fully offline**: local `.pth` plus `data/remedies.json`, no network calls.
That is the paper's requirement — farmers work where connectivity is worst.

| Route | Purpose |
|---|---|
| `GET /` | upload page (drag-drop, preview, bundled sample images) |
| `POST /predict` | result page: verdict, confidence meter, top-3, remedy cards |
| `POST /api/predict` | JSON endpoint |
| `GET /health` | model state, for monitoring |

```bash
curl -F "image=@data/raw/test/AppleCedarRust1.JPG" http://127.0.0.1:5000/api/predict
```

Command-line inference:

```bash
python -m src.predict --image "data/raw/test/AppleCedarRust1.JPG"
python -m src.predict --dir data/raw/test --json
```

The app loads the model once at first request and caches it. **Restart the app
after retraining** to pick up new weights.

### Notes for real deployment

- `app.py` uses Flask's development server. For anything public use a WSGI server
  (`waitress-serve --port=5000 app.app:app` on Windows, gunicorn on Linux).
- It binds to `127.0.0.1` (local only). To expose it on your network, change
  `host` to `0.0.0.0` in `app.py` — understand the security implications first.
- Upload limit is 10 MB, set by `MAX_UPLOAD_BYTES` in `config.py`.
- Low-confidence advisory fires below 60% (`LOW_CONFIDENCE_THRESHOLD`).

---

## 7. Full command reference, in order

```bash
# ---- setup (once per machine) ----
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/check_env.py                      # MUST say CUDA available: True
python scripts/prepare_dataset.py --zip "path/to/archive.zip"

# ---- verify before training ----
python -m pytest tests/ -v                       # 32 tests

# ---- train ----
python -m src.train --img-size 256 --epochs 5 --batch-size 64 --amp

# ---- evaluate + figures ----
python -m src.evaluate
python scripts/make_sample_grid.py
python -m src.compare_models
python -m src.lr_sweep

# ---- use ----
python -m src.predict --image "data/raw/test/AppleCedarRust1.JPG"
python app/app.py
```

---

## 8. What was done (full progress log)

### Starting point

The original project code **did not survive**. The two files believed to be it —
`random_forest.py` and `Random_Forest.ipynb` — are a malicious-URL classifier
(scikit-learn RandomForest over 79 URL features, target `URL_Type_obf_Type`), an
entirely different project. There was no plant-disease code at all.

The rebuild was therefore driven from `final_research_paper_ASEP_new.docx` and
`ASEP_Endsem1_march[1][1].pptx`, which together specify the architecture,
hyperparameters, dataset, deployment stack and remedy approach.

Conflict resolved: the December 2024 decks say *Streamlit + EfficientNetB0*, but
the final paper and the March 2025 deck both say *Flask + ResNet9*. The later
sources won.

### Built

| Component | Notes |
|---|---|
| `scripts/prepare_dataset.py` | threaded extraction, skips the case-duplicate, verifies counts |
| `scripts/check_env.py` | device report, catches the CPU-wheel trap, recommends settings |
| `scripts/make_sample_grid.py` | dataset figures |
| `src/model.py` | ResNet9 (6.59M) + ResNet6 baseline (0.68M) |
| `src/data.py` | transforms, loaders, device helpers, stratified subsetting |
| `src/engine.py` | One-Cycle loop, gradient clipping, AMP |
| `src/train.py` | training CLI, per-epoch checkpointing |
| `src/evaluate.py` | metrics, confusion matrix, figures |
| `src/inference.py` | shared model loading + prediction |
| `src/predict.py` | inference CLI |
| `src/remedies.py` + `data/remedies.json` | 38-class offline remedy database |
| `src/compare_models.py`, `src/lr_sweep.py` | paper comparison artifacts |
| `src/viz.py` | shared figure styling, colourblind-validated palette |
| `app/` | Flask app, templates, CSS |
| `tests/test_smoke.py` | 32 tests |
| `notebooks/ResNet9_PlantDisease_Colab.ipynb` | GPU backup path |

### Verified by running, not by assumption

- Dataset extraction: 2.9 min, 505 files/s, exact counts confirmed
- CLI inference: Cedar Apple Rust at 92.1% confidence, 129 ms
- All four Flask routes with real images
- Templates render correctly in healthy / diseased / low-confidence states
- 32/32 tests pass
- Every figure inspected visually, not just generated

### Bugs found and fixed during the build

1. **BatchNorm crash on a 1-sample final batch.** The first training run died at
   the end of epoch 1: 21,089 images / batch 32 leaves a remainder of *exactly 1*,
   and BatchNorm cannot process a single sample in training mode at 64px, where the
   deepest feature map is already 1x1. Fixed by `should_drop_last()` in `src/data.py`.
2. **`Path` not JSON-serializable** — would have made `/health` and `metrics.json`
   fail the moment a model existed. Fixed with `_json_safe()`.
3. **Chart titles silently blanked** — `ax.get_title()` reads the *center* title,
   so reading back a left-aligned one returned an empty string.
4. **Title/subtitle collision** on tall figures, from axes-fraction offsets.
5. **History lost on interruption** — now written after every epoch.
6. **tqdm bloating detached logs** — `disable=None` auto-disables when not a TTY.

### Deliberate deviations from the paper

- **`AdaptiveMaxPool2d(1)` instead of a fixed `MaxPool2d(4)`** in the classifier.
  Numerically identical at 256px, but the fixed version *crashes* at 64px because
  the feature map reaches 1x1 before the final pool. This is what makes CPU
  training at 64/128px possible from the same weights-compatible definition.
- **OpenCV is not used.** The paper lists it; Pillow and torchvision cover all
  preprocessing, so it is not a dependency.
- **Transforms are defined once** in `src/data.py` and imported by training,
  evaluation, the CLI and the app. Inference preprocessing drifting from training
  preprocessing is the most common silent accuracy killer in projects like this.

### Corrections to the research paper

1. **The mildew confusion claim does not reproduce.** Measured: none observed. The
   real top error is Tomato Late Blight -> Potato Late Blight (same pathogen).
2. **The averaging method is unstated.** The paper reports P 0.94 / R 0.96 /
   F1 0.92. That is *not* self-contradictory, despite appearances: macro-F1 is the
   mean of per-class F1 scores, not the harmonic mean of macro-P and macro-R, so it
   can legitimately fall below both. What the paper omits is which averaging it
   used. `evaluate.py` reports macro and weighted side by side.
3. **The optimal learning rate is disputed.** The paper says 0.1; the canonical
   ResNet9 implementation uses 0.01, which is what achieved 98.36% here.
   `src/lr_sweep.py` settles it empirically — run it to get a citable answer.

---

## 9. Gotchas

| Symptom | Cause and fix |
|---|---|
| Extraction takes hours | Using File Explorer. Use `scripts/prepare_dataset.py`. |
| `bash: Engineering: command not found` | Unquoted path — the `&` in the folder name splits the command. Quote it. |
| Training 20x slower than expected on the GPU box | CPU-only torch wheel. Run `check_env.py`; reinstall from the cu1xx index. |
| `CUDA out of memory` | Halve `--batch-size`, then lower `--img-size`. |
| `Expected more than 1 value per channel` | Final batch of 1 sample. Already handled by `should_drop_last`. |
| Detached run writes an empty log | Missing `-u`. Python buffers stdout when redirected. |
| App shows "No trained model yet" | Expected before training. Run `src.train`. |
| App still shows the old accuracy | It caches the model at first request. Restart it. |
| Predictions look random | The checkpoint's `img_size` must match training — it is stored and applied automatically. Retrain if you edited transforms. |

---

## 10. Reference

- Dataset: <https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset>
  (87,900 images, 38 classes, 70,295 train / 17,572 valid / 33 test)
- Architecture: ResNet9 — conv1(3->64), conv2(64->128, pool), res1, conv3(128->256,
  pool), conv4(256->512, pool), res2, AdaptiveMaxPool -> Linear(512, 38)
- Project: Applied Science & Engineering Project (ASEP), Computer Science
  Engineering, Vishwakarma Institute of Technology, Pune. Guide: Prof. Vijay U. Rathod.

Remedy guidance in `data/remedies.json` is compiled from standard agricultural
extension sources and is general advice only — follow local label rates and consult
an extension officer before applying any chemical treatment.

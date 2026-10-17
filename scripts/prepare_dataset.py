"""Extract the PlantVillage dataset from the Kaggle zip - fast and correctly.

Why this script exists instead of just unzipping in Explorer:

The archive contains 175,767 entries, which is the entire dataset TWICE:

    New Plant Diseases Dataset(Augmented)/...   87,867 files, 1.43 GB
    new plant diseases dataset(augmented)/...   87,867 files, 1.43 GB  <- duplicate
    test/...                                        33 files

Those two folder names differ only in case. Linux treats them as two distinct
directories; Windows is case-insensitive, so Explorer writes the second copy
over the first - 87,867 overwrite collisions, one per file - on top of its
single-threaded zip handler and Defender scanning all 175k files. That is what
turns a few minutes of work into hours.

This script extracts only the canonical subtree plus the test images, flattens
the doubled nesting, and uses a thread pool (zlib releases the GIL, so threads
genuinely help on an NVMe SSD). It is resumable: re-running skips files already
present at the correct size.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --zip "D:/downloads/archive.zip" --workers 8
    python scripts/prepare_dataset.py --per-class 300      # quick subset
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it


_local = threading.local()


def _handle(zip_path):
    """One ZipFile per thread - a single handle is not thread-safe to share."""
    zf = getattr(_local, "zf", None)
    if zf is None:
        zf = zipfile.ZipFile(zip_path)
        _local.zf = zf
    return zf


def plan_extraction(zip_path, per_class):
    """Map zip entry names to flattened output paths.

    Returns (jobs, skipped) where jobs is a list of
    (zip_name, relative_output_path, uncompressed_size).
    """
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]

    canonical = config.ZIP_CANONICAL_PREFIX
    jobs = []
    duplicates = 0

    # Bucket train/valid entries by class so --per-class can trim them.
    buckets = collections.defaultdict(list)

    for info in infos:
        name = info.filename
        if name.startswith(canonical):
            rel = name[len(canonical):]
            parts = rel.split("/")
            if len(parts) < 3:
                continue
            split, cls = parts[0], parts[1]
            if split not in ("train", "valid"):
                continue
            out = split + "/" + cls + "/" + parts[-1]
            buckets[(split, cls)].append((name, out, info.file_size))
        elif name.startswith(config.ZIP_TEST_PREFIX):
            # Entries are "test/test/<file>" - flatten to "test/<file>".
            jobs.append((name, "test/" + Path(name).name, info.file_size))
        else:
            # The lowercase duplicate copy, plus any stray metadata.
            duplicates += 1

    for (split, _cls), items in buckets.items():
        items.sort(key=lambda t: t[1])
        if per_class is not None:
            # valid is ~25% the size of train; keep the ratio sane.
            limit = per_class if split == "train" else max(1, per_class // 4)
            items = items[:limit]
        jobs.extend(items)

    jobs.sort(key=lambda t: t[1])
    return jobs, duplicates


def extract(zip_path, out_dir, jobs, workers):
    # Pre-create every directory single-threaded to avoid mkdir races.
    parents = set()
    for _name, out, _size in jobs:
        parents.add(str(Path(out).parent))
    for parent in parents:
        (out_dir / parent).mkdir(parents=True, exist_ok=True)

    counters = {"written": 0, "skipped": 0}
    lock = threading.Lock()

    def work(job):
        name, rel, size = job
        target = out_dir / rel
        # Resumable: trust an existing file only if its size matches exactly.
        try:
            if target.stat().st_size == size:
                with lock:
                    counters["skipped"] += 1
                return
        except OSError:
            pass
        zf = _handle(zip_path)
        with zf.open(name) as src:
            data = src.read()
        tmp = target.with_suffix(target.suffix + ".part")
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
        with lock:
            counters["written"] += 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in tqdm(pool.map(work, jobs), total=len(jobs),
                      desc="extracting", unit="file", smoothing=0.05):
            pass
    return counters["written"], counters["skipped"], time.time() - t0


def verify(out_dir, strict):
    print()
    print("=" * 68)
    print("VERIFICATION")
    print("=" * 68)

    ok = True
    counts = {}
    for split, expected in (("train", config.EXPECTED_TRAIN),
                            ("valid", config.EXPECTED_VALID)):
        d = out_dir / split
        if not d.is_dir():
            print("  [FAIL] missing split directory: " + str(d))
            ok = False
            continue
        c = collections.Counter()
        for cls_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            c[cls_dir.name] = sum(1 for f in cls_dir.iterdir() if f.is_file())
        counts[split] = c
        total = sum(c.values())
        n_cls = len(c)
        print("  %-6s classes=%3d (expect %d)  images=%6d (expect %d)"
              % (split, n_cls, config.NUM_CLASSES, total, expected))
        if n_cls != config.NUM_CLASSES:
            print("  [FAIL] %s: expected %d classes, found %d"
                  % (split, config.NUM_CLASSES, n_cls))
            ok = False
        if strict and total != expected:
            print("  [FAIL] %s: expected %d images, found %d" % (split, expected, total))
            ok = False

    test_dir = out_dir / "test"
    n_test = sum(1 for f in test_dir.iterdir() if f.is_file()) if test_dir.is_dir() else 0
    print("  %-6s %22s images=%6d (expect %d)"
          % ("test", "", n_test, config.EXPECTED_TEST))
    if strict and n_test != config.EXPECTED_TEST:
        print("  [WARN] test: expected %d images, found %d" % (config.EXPECTED_TEST, n_test))

    if "train" in counts:
        print()
        print("Per-class image counts (regenerates the paper's dataset table):")
        print("  %-52s %7s %7s" % ("Plant/Disease Class", "train", "valid"))
        print("  " + "-" * 68)
        valid_counts = counts.get("valid", collections.Counter())
        for cls in sorted(counts["train"]):
            print("  %-52s %7d %7d" % (cls, counts["train"][cls], valid_counts.get(cls, 0)))
        print("  " + "-" * 68)
        print("  %-52s %7d %7d" % ("TOTAL", sum(counts["train"].values()),
                                   sum(valid_counts.values())))

    print()
    print("  RESULT: " + ("OK" if ok else "PROBLEMS FOUND"))
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Extract the PlantVillage dataset from the Kaggle zip.")
    ap.add_argument("--zip", type=Path, default=config.DEFAULT_ZIP,
                    help="source archive (default: %s)" % config.DEFAULT_ZIP)
    ap.add_argument("--out", type=Path, default=config.RAW_DIR,
                    help="output directory (default: %s)" % config.RAW_DIR)
    ap.add_argument("--workers", type=int, default=8, help="extraction threads (default 8)")
    ap.add_argument("--per-class", type=int, default=None,
                    help="extract at most N train images per class (valid gets N//4). "
                         "Default: everything.")
    args = ap.parse_args()

    if not args.zip.is_file():
        print("ERROR: archive not found: " + str(args.zip), file=sys.stderr)
        print("Download it from "
              "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset",
              file=sys.stderr)
        return 2

    size_gb = args.zip.stat().st_size / 1e9
    print("=" * 68)
    print("PlantVillage dataset preparation")
    print("=" * 68)
    print("  archive : %s  (%.2f GB)" % (args.zip, size_gb))
    print("  output  : %s" % args.out)
    print("  workers : %d" % args.workers)
    print()
    print("  TIP: Windows Defender scans every extracted file. To speed this up,")
    print("  run this in an ADMIN PowerShell first:")
    print('    Add-MpPreference -ExclusionPath "%s"' % args.out)
    print()

    print("Reading archive index...")
    jobs, duplicates = plan_extraction(args.zip, args.per_class)
    total_bytes = sum(s for _n, _o, s in jobs)
    print("  %s files to extract (%.2f GB)" % (format(len(jobs), ","), total_bytes / 1e9))
    print("  %s entries skipped (the case-duplicated second copy)"
          % format(duplicates, ","))
    if not jobs:
        print("ERROR: nothing matched the expected layout. Is this the right archive?",
              file=sys.stderr)
        return 2
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    written, skipped, elapsed = extract(args.zip, args.out, jobs, args.workers)
    rate = ("  (%.0f files/s)" % (written / elapsed)) if elapsed > 0 and written else ""
    print("  wrote %s files, skipped %s already present, in %.1f min%s"
          % (format(written, ","), format(skipped, ","), elapsed / 60, rate))

    ok = verify(args.out, strict=args.per_class is None)
    if ok:
        print()
        print("Next: python -m src.train --img-size 64 --subset 0.1 --epochs 1")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

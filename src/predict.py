"""Command-line inference.

Examples:
    python -m src.predict --image "data/raw/test/AppleCedarRust1.JPG"
    python -m src.predict --dir data/raw/test
    python -m src.predict --dir data/raw/test --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.inference import ModelNotTrained, load_image, load_model, predict_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def bar(fraction: float, width: int = 28) -> str:
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


def report(path: Path, result: dict, verbose: bool = True) -> None:
    print()
    print("=" * 68)
    print("  %s" % path.name)
    print("=" * 68)
    status = "HEALTHY" if result["healthy"] else "DISEASED"
    print("  prediction : %s" % result["display"])
    print("  status     : %s" % status)
    print("  confidence : %.2f%%" % (result["confidence"] * 100))

    if result["low_confidence"]:
        print()
        print("  [!] LOW CONFIDENCE (below %.0f%%). The paper's error analysis"
              % (config.LOW_CONFIDENCE_THRESHOLD * 100))
        print("      flags poor lighting and leaves occluded by other plant")
        print("      parts. Retake the photo: fill the frame with one leaf,")
        print("      in even daylight, against a plain background.")

    print()
    print("  top %d:" % len(result["predictions"]))
    for p in result["predictions"]:
        print("    %-44s %s %6.2f%%"
              % (p["display"][:44], bar(p["confidence"]), p["confidence"] * 100))

    if not verbose:
        return

    rem = result["remedy"]
    print()
    if rem["pathogen"]:
        print("  cause      : %s" % rem["pathogen"])
    print("  symptoms   : %s" % rem["symptoms"])
    print()
    print("  ORGANIC    : %s" % rem["organic"])
    print("  CHEMICAL   : %s" % rem["chemical"])
    print("  PREVENTION : %s" % rem["prevention"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Classify plant leaf images.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="a single image file")
    src.add_argument("--dir", type=Path, help="a directory of images")
    ap.add_argument("--model", type=Path, default=config.MODEL_FILE)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    ap.add_argument("--quiet", action="store_true", help="omit remedy text")
    args = ap.parse_args(argv)

    try:
        model, class_names, img_size = load_model(args.model)
    except ModelNotTrained as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    if args.image:
        paths = [args.image]
    else:
        paths = sorted(p for p in args.dir.iterdir()
                       if p.is_file() and p.suffix in IMAGE_SUFFIXES)
    if not paths:
        print("ERROR: no images found", file=sys.stderr)
        return 2

    missing = [p for p in paths if not p.is_file()]
    if missing:
        print("ERROR: file not found: %s" % missing[0], file=sys.stderr)
        return 2

    if not args.json:
        print("model: %s  (%s, %dpx, %d classes)"
              % (args.model.name, "resnet9", img_size, len(class_names)))

    results = []
    t0 = time.time()
    for path in paths:
        result = predict_image(model, load_image(path), class_names, img_size,
                               topk=args.topk)
        results.append({"file": path.name, **{k: v for k, v in result.items()
                                              if k != "remedy"},
                        "remedy": result["remedy"]})
        if not args.json:
            report(path, result, verbose=not args.quiet)
    elapsed = time.time() - t0

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print()
        print("-" * 68)
        print("  %d image(s) in %.2f s  (%.0f ms each)"
              % (len(paths), elapsed, 1000 * elapsed / len(paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

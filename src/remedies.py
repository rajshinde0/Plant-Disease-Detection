"""Offline remedy lookup.

The paper's reasoning for keeping this local rather than querying an online
service: farmers often work where connectivity is poor or absent, so a remedy
that needs the internet is a remedy that is unavailable exactly when it is
needed. Everything here reads from data/remedies.json - no network calls.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


@lru_cache(maxsize=1)
def load_remedies(path: str | None = None) -> dict:
    """Load the remedy table, excluding the _meta block."""
    p = Path(path) if path else config.REMEDIES_FILE
    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def load_meta(path: str | None = None) -> dict:
    p = Path(path) if path else config.REMEDIES_FILE
    with open(p, encoding="utf-8") as fh:
        return json.load(fh).get("_meta", {})


def prettify(class_name: str) -> str:
    """Fallback display name, derived from a raw class-folder name.

    Handles the dataset's quirks: the '___' crop/disease separator, trailing
    underscores ('Common_rust_'), and embedded spaces.
    """
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    else:
        crop, disease = class_name, ""
    crop = crop.replace("_", " ").strip().strip(",")
    disease = disease.replace("_", " ").strip()
    if not disease:
        return crop
    return "%s - %s" % (crop, disease[:1].upper() + disease[1:])


def lookup(class_name: str) -> dict:
    """Return the remedy record for a class name.

    Unknown classes get a safe placeholder rather than raising, so a model
    trained on a different class list cannot crash the web app.
    """
    remedies = load_remedies()
    rec = remedies.get(class_name)
    if rec is None:
        return {
            "crop": prettify(class_name).split(" - ")[0],
            "disease": prettify(class_name),
            "healthy": class_name.lower().endswith("healthy"),
            "pathogen": None,
            "symptoms": "No remedy record found for this class.",
            "organic": "Not available offline for this class.",
            "chemical": "Not available offline for this class.",
            "prevention": "Consult your local agricultural extension officer.",
            "display": prettify(class_name),
            "known": False,
        }
    out = dict(rec)
    out["display"] = "%s - %s" % (rec["crop"], rec["disease"])
    out["known"] = True
    return out


def display_name(class_name: str) -> str:
    return lookup(class_name)["display"]


def is_healthy(class_name: str) -> bool:
    return bool(lookup(class_name)["healthy"])


def coverage(class_names) -> tuple[list, list]:
    """(missing, extra) relative to a model's class list. Used by the tests."""
    keys = set(load_remedies())
    names = set(class_names)
    return sorted(names - keys), sorted(keys - names)

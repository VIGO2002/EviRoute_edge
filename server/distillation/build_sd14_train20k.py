#!/usr/bin/env python3

from pathlib import Path
import hashlib
import random
import os
import json

import pandas as pd
import torch


ROOT = Path("/root/autodl-tmp/EviRoute_edge")

OLD_FEATURES = Path(
    "/root/autodl-tmp/FGTS-main/"
    "features/genimage_sd14_smoke2k_mid20_patches"
)

EVAL = ROOT / "evaluation/genimage_8domain_eval8000/manifest.csv"

OUT = ROOT / "distillation/data_manifests"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261008
TARGET_PER_CLASS = 10000

CLASS_NAMES = {
    0: "nature",
    1: "ai",
}

EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".webp", ".bmp", ".tif", ".tiff"
}


def canonical(p):
    return str(Path(str(p)).resolve())


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            h.update(data)

    return h.hexdigest()


def load_split(name):
    obj = torch.load(
        OLD_FEATURES / f"{name}.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    paths = [canonical(p) for p in obj["paths"]]
    labels = obj["labels"].cpu().numpy().astype(int)

    assert len(paths) == len(labels)
    assert len(set(paths)) == len(paths)
    assert set(labels.tolist()) == {0, 1}

    return pd.DataFrame({
        "path": paths,
        "label": labels,
    })


def find_train_root(path):
    p = Path(path)

    for parent in p.parents:
        if parent.name == "train":
            return parent

    raise RuntimeError(
        f"Cannot identify training root: {path}"
    )


def list_candidates(folder, excluded):
    candidates = []

    for base, _, files in os.walk(folder):
        for filename in files:
            path = Path(base) / filename

            if path.suffix.lower() not in EXTENSIONS:
                continue

            full = canonical(path)

            if full not in excluded:
                candidates.append(full)

    return sorted(candidates)


def main():
    random.seed(SEED)

    historical = {
        name: load_split(name)
        for name in ("train", "val", "test")
    }

    original = historical["train"]

    benchmark = pd.read_csv(EVAL)
    assert len(benchmark) == 8000

    eval_paths = set(
        benchmark["path"].map(canonical)
    )

    val_paths = set(historical["val"]["path"])
    test_paths = set(historical["test"]["path"])
    train_paths = set(original["path"])

    heldout = val_paths | test_paths | eval_paths

    assert not (train_paths & heldout)
    assert not (val_paths & test_paths)

    excluded = train_paths | heldout

    train_root = find_train_root(
        original["path"].iloc[0]
    )

    print("=" * 72)
    print("GENIMAGE SD1.4 20K TRAIN MANIFEST")
    print("=" * 72)
    print("Seed:", SEED)
    print("Train root:", train_root)
    print("Historical train:", len(train_paths))
    print("Held-out unique paths:", len(heldout))

    rows = []

    for label, classname in CLASS_NAMES.items():

        old_paths = original.loc[
            original["label"] == label, "path"
        ].tolist()

        folder = train_root / classname

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        required = TARGET_PER_CLASS - len(old_paths)

        if required <= 0:
            raise RuntimeError(
                f"Unexpected historical class count: {classname}"
            )

        candidates = list_candidates(
            folder, excluded
        )

        if len(candidates) < required:
            raise RuntimeError(
                f"Not enough samples for {classname}"
            )

        rng = random.Random(SEED + label)

        selected = rng.sample(
            candidates, required
        )

        for path in old_paths:
            rows.append({
                "path": path,
                "label": label,
                "class_name": classname,
                "origin": "original_2800",
            })

        for path in selected:
            rows.append({
                "path": path,
                "label": label,
                "class_name": classname,
                "origin": "new",
            })

        print()
        print("Class:", classname)
        print("Historical:", len(old_paths))
        print("New:", len(selected))
        print("Total:", len(old_paths) + len(selected))

    rng = random.Random(SEED)
    rng.shuffle(rows)

    df = pd.DataFrame(rows)

    assert len(df) == 20000
    assert df["label"].value_counts().to_dict() == {
        0: 10000, 1: 10000
    }
    assert df["path"].nunique() == len(df)

    assert not (set(df["path"]) & heldout)

    print()
    print("=" * 72)
    print("CONTENT HASH AUDIT")
    print("=" * 72)

    # Hash all held-out files and all selected training files.
    # This detects byte-identical duplicates at different paths.
    heldout_hashes = {}

    heldout_ordered = sorted(heldout)

    for i, path in enumerate(heldout_ordered, 1):
        digest = sha256(path)
        heldout_hashes.setdefault(digest, path)

        if i % 2000 == 0:
            print(
                f"Held-out hashed: {i}/{len(heldout_ordered)}",
                flush=True,
            )

    train_hashes = {}
    duplicates = []

    for i, path in enumerate(df["path"], 1):

        digest = sha256(path)

        if digest in heldout_hashes:
            duplicates.append({
                "kind": "train_vs_heldout",
                "train_path": path,
                "other_path": heldout_hashes[digest],
            })

        if digest in train_hashes:
            duplicates.append({
                "kind": "within_train",
                "train_path": path,
                "other_path": train_hashes[digest],
            })
        else:
            train_hashes[digest] = path

        if i % 4000 == 0:
            print(
                f"Training hashed: {i}/{len(df)}",
                flush=True,
            )

    if duplicates:
        diagnostic = OUT / "train20k_duplicates.csv"
        pd.DataFrame(duplicates).to_csv(
            diagnostic,
            index=False,
        )

        raise RuntimeError(
            f"Detected {len(duplicates)} exact-content "
            f"duplicates. See {diagnostic}"
        )

    output = OUT / "sd14_train20k_seed20261008.csv"

    df.to_csv(output, index=False)

    summary = {
        "seed": SEED,
        "samples": len(df),
        "class_counts": {
            str(k): int(v)
            for k, v in df["label"].value_counts().items()
        },
        "original_count": int(
            (df["origin"] == "original_2800").sum()
        ),
        "new_count": int(
            (df["origin"] == "new").sum()
        ),
        "path_overlap_with_heldout": 0,
        "byte_identical_duplicate_count": 0,
        "content_hash": "SHA256",
        "manifest": str(output),
    }

    (OUT / "sd14_train20k_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Manifest:", output)
    print("Original samples:", summary["original_count"])
    print("New samples:", summary["new_count"])
    print("Path overlap: 0")
    print("Byte-identical duplicates: 0")
    print()
    print("SD1.4 TRAIN20K MANIFEST: PASS")


if __name__ == "__main__":
    main()

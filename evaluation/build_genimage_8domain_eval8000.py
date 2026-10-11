from pathlib import Path
import json

import numpy as np
import pandas as pd


DATA_ROOT = Path("/root/autodl-tmp/GenImage_extracted")
PROJECT = Path("/root/autodl-tmp/EviRoute_edge")

OUT = (
    PROJECT
    / "evaluation"
    / "genimage_8domain_eval8000"
)

OLD_SPLIT = (
    PROJECT
    / "reference"
    / "splits"
    / "genimage_sd14_smoke2k_vitb_mid6_split_manifest.csv"
)

SEED = 20261007

N_REAL = 500
N_FAKE = 500


DOMAINS = {
    "ADM": {
        "val": (
            DATA_ROOT
            / "ADM"
            / "imagenet_ai_0508_adm"
            / "val"
        ),
        "role": "unseen",
    },

    "BigGAN": {
        "val": (
            DATA_ROOT
            / "BigGAN"
            / "imagenet_ai_0419_biggan"
            / "val"
        ),
        "role": "unseen",
    },

    "Midjourney": {
        "val": (
            DATA_ROOT
            / "Midjourney"
            / "imagenet_midjourney"
            / "val"
        ),
        "role": "unseen",
    },

    "VQDM": {
        "val": (
            DATA_ROOT
            / "VQDM"
            / "imagenet_ai_0419_vqdm"
            / "val"
        ),
        "role": "unseen",
    },

    "GLIDE": {
        "val": (
            DATA_ROOT
            / "glide"
            / "imagenet_glide"
            / "val"
        ),
        "role": "unseen",
    },

    "SD1.4": {
        "val": (
            DATA_ROOT
            / "stable_diffusion_v_1_4"
            / "imagenet_ai_0419_sdv4"
            / "val"
        ),
        "role": "in_domain",
    },

    "SD1.5": {
        "val": (
            DATA_ROOT
            / "stable_diffusion_v_1_5"
            / "imagenet_ai_0424_sdv5"
            / "val"
        ),
        "role": "unseen",
    },

    "Wukong": {
        "val": (
            DATA_ROOT
            / "wukong"
            / "imagenet_ai_0424_wukong"
            / "val"
        ),
        "role": "unseen",
    },
}


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_images(directory):
    return sorted(
        [
            p
            for p in directory.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() in IMAGE_EXTS
            )
        ],
        key=lambda p: p.name,
    )


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        SEED
    )

    print("=" * 80)
    print("BUILD GENIMAGE 8-DOMAIN EVAL-8000")
    print("=" * 80)

    print(
        f"Per domain: "
        f"{N_REAL} REAL + {N_FAKE} FAKE"
    )

    print(
        "REAL protocol: shared SD1.4 val/nature subset"
    )

    print(
        "Seed:",
        SEED
    )

    # ------------------------------------------------------------------
    # 1. Historical split exclusion
    # ------------------------------------------------------------------

    old_real_basenames = set()
    old_sd14_paths = set()

    if OLD_SPLIT.exists():

        old_df = pd.read_csv(
            OLD_SPLIT
        )

        for _, row in old_df.iterrows():

            p = Path(
                str(row["path"])
            )

            old_sd14_paths.add(
                str(p.resolve())
            )

            if int(row["label"]) == 0:
                old_real_basenames.add(
                    p.name
                )

        print()
        print(
            "Historical split rows:",
            len(old_df)
        )

        print(
            "Historical REAL basenames excluded:",
            len(old_real_basenames)
        )

    else:

        print()
        print(
            "WARNING: historical split not found"
        )

    # ------------------------------------------------------------------
    # 2. Build one shared REAL reference subset
    # ------------------------------------------------------------------

    shared_real_dir = (
        DOMAINS["SD1.4"]["val"]
        / "nature"
    )

    if not shared_real_dir.exists():
        raise FileNotFoundError(
            shared_real_dir
        )

    real_candidates = get_images(
        shared_real_dir
    )

    print()
    print(
        "Shared REAL candidates before exclusion:",
        len(real_candidates)
    )

    real_candidates = [
        p
        for p in real_candidates
        if p.name not in old_real_basenames
    ]

    print(
        "Shared REAL candidates after exclusion :",
        len(real_candidates)
    )

    if len(real_candidates) < N_REAL:
        raise RuntimeError(
            "Insufficient shared REAL candidates"
        )

    real_indices = rng.choice(
        len(real_candidates),
        size=N_REAL,
        replace=False,
    )

    shared_reals = [
        real_candidates[
            int(i)
        ]
        for i in real_indices
    ]

    shared_reals = sorted(
        shared_reals,
        key=lambda p: p.name,
    )

    print(
        "Selected shared REAL:",
        len(shared_reals)
    )

    # ------------------------------------------------------------------
    # 3. Build per-domain 500 REAL + 500 FAKE
    # ------------------------------------------------------------------

    rows = []

    global_index = 0

    domain_summary = {}

    for domain, cfg in DOMAINS.items():

        val_root = cfg["val"]
        fake_dir = (
            val_root / "ai"
        )

        if not fake_dir.exists():
            raise FileNotFoundError(
                fake_dir
            )

        fake_candidates = get_images(
            fake_dir
        )

        print()
        print("-" * 80)
        print("DOMAIN:", domain)

        print(
            "FAKE candidates before exclusion:",
            len(fake_candidates)
        )

        # SD1.4 only:
        # do not reuse exact historical smoke-split samples.
        if domain == "SD1.4":

            fake_candidates = [
                p
                for p in fake_candidates
                if (
                    str(p.resolve())
                    not in old_sd14_paths
                )
            ]

            print(
                "FAKE candidates after exclusion :",
                len(fake_candidates)
            )

        if len(fake_candidates) < N_FAKE:
            raise RuntimeError(
                f"{domain}: insufficient FAKE candidates"
            )

        fake_indices = rng.choice(
            len(fake_candidates),
            size=N_FAKE,
            replace=False,
        )

        selected_fakes = [
            fake_candidates[
                int(i)
            ]
            for i in fake_indices
        ]

        selected_fakes = sorted(
            selected_fakes,
            key=lambda p: p.name,
        )

        domain_index = 0

        # --------------------------------------------------------------
        # Shared REAL
        # --------------------------------------------------------------

        for p in shared_reals:

            rows.append({
                "global_index": global_index,
                "domain_index": domain_index,
                "domain": domain,
                "domain_role": cfg["role"],
                "label": 0,
                "class_name": "real",
                "path": str(
                    p.resolve()
                ),
                "filename": p.name,
                "real_source": "SD1.4_val_nature_shared",
            })

            global_index += 1
            domain_index += 1

        # --------------------------------------------------------------
        # Domain FAKE
        # --------------------------------------------------------------

        for p in selected_fakes:

            rows.append({
                "global_index": global_index,
                "domain_index": domain_index,
                "domain": domain,
                "domain_role": cfg["role"],
                "label": 1,
                "class_name": "fake",
                "path": str(
                    p.resolve()
                ),
                "filename": p.name,
                "real_source": "",
            })

            global_index += 1
            domain_index += 1

        domain_summary[
            domain
        ] = {
            "role": cfg["role"],
            "real": N_REAL,
            "fake": N_FAKE,
            "total": N_REAL + N_FAKE,
        }

        print(
            f"Selected: "
            f"{N_REAL} REAL + {N_FAKE} FAKE"
        )

    # ------------------------------------------------------------------
    # 4. Validate manifest
    # ------------------------------------------------------------------

    df = pd.DataFrame(
        rows
    )

    expected = (
        len(DOMAINS)
        * (N_REAL + N_FAKE)
    )

    if len(df) != expected:
        raise RuntimeError(
            f"Expected {expected}, got {len(df)}"
        )

    if not df[
        "global_index"
    ].is_unique:
        raise RuntimeError(
            "global_index is not unique"
        )

    for domain in DOMAINS:

        sub = df[
            df["domain"] == domain
        ]

        n_real = int(
            (sub["label"] == 0).sum()
        )

        n_fake = int(
            (sub["label"] == 1).sum()
        )

        if (
            n_real != N_REAL
            or n_fake != N_FAKE
        ):
            raise RuntimeError(
                f"{domain}: "
                f"REAL={n_real}, FAKE={n_fake}"
            )

    missing = [
        p
        for p in df["path"]
        if not Path(p).exists()
    ]

    if missing:
        raise RuntimeError(
            f"Missing paths: {missing[:5]}"
        )

    # Verify every domain uses exactly the same 500 REAL paths.
    reference_real_paths = None

    for domain in DOMAINS:

        current = set(
            df[
                (df["domain"] == domain)
                & (df["label"] == 0)
            ]["path"]
        )

        if reference_real_paths is None:
            reference_real_paths = current

        elif current != reference_real_paths:
            raise RuntimeError(
                f"{domain}: shared REAL subset mismatch"
            )

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------

    manifest_path = (
        OUT / "manifest.csv"
    )

    df.to_csv(
        manifest_path,
        index=False,
    )

    with open(
        OUT / "shared_real_paths.txt",
        "w",
        encoding="utf-8",
    ) as f:

        for p in shared_reals:
            f.write(
                str(p.resolve())
                + "\n"
            )

    summary = {
        "benchmark": "GenImage-8 Eval8000",
        "seed": SEED,
        "total_samples": len(df),
        "samples_per_domain": 1000,
        "real_per_domain": N_REAL,
        "fake_per_domain": N_FAKE,
        "shared_real_protocol": True,
        "shared_real_source": (
            "SD1.4/imagenet_ai_0419_sdv4/"
            "val/nature"
        ),
        "historical_sd14_overlap_exclusion": (
            OLD_SPLIT.exists()
        ),
        "domains": domain_summary,
        "primary_metrics": [
            "Unseen-7 Macro AUC",
            "Unseen-7 Worst AUC",
            "Unseen-7 Macro AP",
            "Unseen-7 Macro ACC",
        ],
        "secondary_metrics": [
            "SD1.4 AUC/AP/ACC",
            "8-domain Macro AUC",
            "8-domain Macro AP",
            "8-domain Macro ACC",
        ],
    }

    with open(
        OUT / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 6. Final report
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        "Total samples:",
        len(df)
    )

    print(
        "Shared REAL subset:",
        "PASS"
    )

    print(
        "All paths valid:",
        "PASS"
    )

    print()

    for domain in DOMAINS:

        sub = df[
            df["domain"] == domain
        ]

        print(
            f"{domain:12s}"
            f" role={DOMAINS[domain]['role']:9s}"
            f" real={(sub['label'] == 0).sum():4d}"
            f" fake={(sub['label'] == 1).sum():4d}"
            f" total={len(sub):4d}"
        )

    print()
    print(
        "Manifest:",
        manifest_path
    )

    print(
        "Summary :",
        OUT / "summary.json"
    )

    print()
    print(
        "GENIMAGE-8 EVAL8000 BUILD: PASS"
    )


if __name__ == "__main__":
    main()

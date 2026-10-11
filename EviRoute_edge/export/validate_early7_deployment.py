from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import timm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.eviroute import EviRouteScorer


CKPT = ROOT / "checkpoints" / "eviroute_vitb_mid6.pt"

REAL_GOLDEN = (
    ROOT
    / "reference"
    / "pytorch_reference"
    / "real_clean_golden.npz"
)

FAKE_GOLDEN = (
    ROOT
    / "reference"
    / "pytorch_reference"
    / "fake_clean_golden.npz"
)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return float("nan")

    return float(np.dot(a, b) / denom)


class EviRouteEarly7Deploy(nn.Module):
    """
    Deployment-oriented EviRoute backbone + patch scorer.

    Exact historical semantics:

        image
          -> DINOv3 ViT-B/16
          -> patch embedding
          -> CLS + 4 register tokens
          -> transformer blocks[0:7]
          -> raw output of blocks[6]
          -> remove 5 prefix tokens
          -> 196 x 768 patch features
          -> FP32 -> FP16 -> FP32 compatibility roundtrip
          -> checkpoint feature normalization
          -> 768 -> 256 -> GELU -> 1 scorer
          -> 196 patch scores

    Top-K aggregation is intentionally NOT inside this module.
    """

    TIMM_NAME = "vit_base_patch16_dinov3"

    def __init__(self, checkpoint_path):
        super().__init__()

        # ----------------------------------------------------------
        # Load DINOv3 ViT-B
        # ----------------------------------------------------------

        bb = timm.create_model(
            self.TIMM_NAME,
            pretrained=True,
            img_size=224,
        )

        if hasattr(bb, "reset_classifier"):
            bb.reset_classifier(0)

        bb.eval()

        if len(bb.blocks) != 12:
            raise RuntimeError(
                f"Expected 12 transformer blocks, "
                f"got {len(bb.blocks)}"
            )

        if int(bb.num_features) != 768:
            raise RuntimeError(
                f"Expected feature_dim=768, "
                f"got {bb.num_features}"
            )

        num_prefix = int(
            getattr(bb, "num_prefix_tokens", 5)
        )

        if num_prefix != 5:
            raise RuntimeError(
                f"Expected 5 prefix tokens, "
                f"got {num_prefix}"
            )

        self.num_prefix_tokens = num_prefix

        # ----------------------------------------------------------
        # Keep only the modules needed through blocks[6]
        # ----------------------------------------------------------

        self.patch_embed = bb.patch_embed

        self.cls_token = bb.cls_token
        self.reg_token = bb.reg_token

        self.pos_embed = bb.pos_embed
        self.pos_drop = bb.pos_drop

        self.patch_drop = getattr(
            bb,
            "patch_drop",
            None,
        )

        self.norm_pre = getattr(
            bb,
            "norm_pre",
            nn.Identity(),
        )

        self.blocks = nn.ModuleList(
            list(bb.blocks[:7])
        )

        # ----------------------------------------------------------
        # Freeze static RoPE for 224x224
        # ----------------------------------------------------------

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                3,
                224,
                224,
                dtype=torch.float32,
            )

            patch_tokens = bb.patch_embed(
                dummy
            )

            native_pos, rope = bb._pos_embed(
                patch_tokens
            )

            manual_pos = self._prefix_embed(
                patch_tokens
            )

            prefix_err = (
                native_pos - manual_pos
            ).abs().max().item()

            print(
                "Prefix embedding max error:",
                prefix_err
            )

            if prefix_err > 1e-7:
                raise RuntimeError(
                    "Manual prefix embedding does not "
                    "match timm _pos_embed"
                )

        self.register_buffer(
            "rope_const",
            rope.detach().clone(),
            persistent=True,
        )

        print(
            "Frozen RoPE shape:",
            tuple(self.rope_const.shape)
        )

        # ----------------------------------------------------------
        # Load EviRoute scorer + feature statistics
        # ----------------------------------------------------------

        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        args = ckpt["args"]

        if int(args["feat_dim"]) != 768:
            raise RuntimeError(
                "Checkpoint feat_dim mismatch"
            )

        if int(args["hidden_dim"]) != 256:
            raise RuntimeError(
                "Checkpoint hidden_dim mismatch"
            )

        if int(args["topk"]) != 10:
            raise RuntimeError(
                "Checkpoint topk mismatch"
            )

        self.scorer = EviRouteScorer(
            feat_dim=768,
            hidden_dim=256,
            dropout=float(args["dropout"]),
        )

        self.scorer.load_state_dict(
            ckpt["model_state_dict"],
            strict=True,
        )

        self.scorer.eval()

        self.register_buffer(
            "mu",
            ckpt["mu"]
            .float()
            .view(1, 1, 768),
        )

        self.register_buffer(
            "std",
            ckpt["std"]
            .float()
            .view(1, 1, 768),
        )

        if torch.any(self.std < 1e-6):
            raise RuntimeError(
                "Invalid std in checkpoint"
            )

        # Historical model is frozen for inference.
        for p in self.parameters():
            p.requires_grad_(False)

        self.eval()

    def _prefix_embed(self, x):
        """
        Static-224 equivalent of timm Eva._pos_embed().

        patch_embed output:
            [B, 14, 14, 768]

        after flatten:
            [B, 196, 768]

        after CLS + 4 register tokens:
            [B, 201, 768]
        """

        if x.ndim != 4:
            raise RuntimeError(
                f"Expected patch_embed output "
                f"[B,H,W,C], got {tuple(x.shape)}"
            )

        B, H, W, C = x.shape

        if (H, W, C) != (14, 14, 768):
            raise RuntimeError(
                f"Expected [B,14,14,768], "
                f"got {tuple(x.shape)}"
            )

        # Exact first operation in Eva._pos_embed()
        # when dynamic_img_size=True.
        x = x.reshape(
            B,
            H * W,
            C,
        )

        cls = self.cls_token.expand(
            B,
            -1,
            -1,
        )

        reg = self.reg_token.expand(
            B,
            -1,
            -1,
        )

        # Current DINOv3-B:
        # no_embed_class=False
        # pos_embed=None
        x = torch.cat(
            [
                cls,
                reg,
                x,
            ],
            dim=1,
        )

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # Match timm _pos_embed() semantics.
        # Normally DINOv3-B inference has no active
        # patch dropout, but handle the module safely.
        if self.patch_drop is not None:

            out = self.patch_drop(x)

            if isinstance(out, tuple):
                x, keep_indices = out

                if keep_indices is not None:
                    raise RuntimeError(
                        "Unexpected active patch dropout "
                        "during deployment inference"
                    )
            else:
                x = out

        return x

    def forward_backbone(self, images):
        """
        Return raw output of blocks[6].
        No final backbone norm.
        """

        x = self.patch_embed(
            images
        )

        x = self._prefix_embed(
            x
        )

        x = self.norm_pre(
            x
        )

        for block in self.blocks:
            x = block(
                x,
                rope=self.rope_const,
            )

        return x

    def forward_debug(self, images):
        """
        Return all important intermediate tensors
        for golden-reference validation.
        """

        full_tokens = self.forward_backbone(
            images
        )

        raw_patches = full_tokens[
            :,
            self.num_prefix_tokens:
            self.num_prefix_tokens + 196,
            :,
        ]

        # Historical compatibility:
        # original extracted features were stored as FP16,
        # then loaded as FP32 for scorer inference.
        legacy_fp16 = raw_patches.to(
            torch.float16
        )

        scorer_input = legacy_fp16.to(
            torch.float32
        )

        normalized = (
            scorer_input - self.mu
        ) / self.std

        patch_scores = (
            self.scorer
            .score_patches(normalized)
        )

        return (
            full_tokens,
            raw_patches,
            legacy_fp16,
            normalized,
            patch_scores,
        )

    def forward(self, images):
        """
        Deployment output:
        [B, 196] patch scores.
        """

        (
            _,
            _,
            _,
            _,
            patch_scores,
        ) = self.forward_debug(images)

        return patch_scores


def compare_tensor(
    name,
    current,
    reference,
):

    a = (
        current
        .detach()
        .cpu()
        .numpy()
    )

    b = np.asarray(reference)

    diff = np.abs(
        a.astype(np.float64)
        - b.astype(np.float64)
    )

    cos = cosine_similarity(
        a,
        b,
    )

    print()
    print(name)
    print(
        "  shape current :",
        a.shape
    )
    print(
        "  shape golden  :",
        b.shape
    )
    print(
        "  cosine        :",
        f"{cos:.10f}"
    )
    print(
        "  mean abs      :",
        f"{diff.mean():.10e}"
    )
    print(
        "  max abs       :",
        f"{diff.max():.10e}"
    )

    return cos, float(diff.max())


def validate_one(
    model,
    golden_path,
    device,
):

    golden = np.load(
        golden_path
    )

    image = torch.from_numpy(
        golden["input_tensor"]
    ).to(
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():

        (
            full_tokens,
            raw_patches,
            legacy_fp16,
            normalized,
            patch_scores,
        ) = model.forward_debug(
            image
        )

        top10_scores, top10_indices = (
            torch.topk(
                patch_scores,
                k=10,
                dim=1,
            )
        )

        image_logit = (
            top10_scores.mean(dim=1)
        )

        probability = torch.sigmoid(
            image_logit
        )

        prediction = (
            probability >= 0.5
        ).long()

    print()
    print("=" * 72)
    print(
        "VALIDATE:",
        golden_path.name
    )
    print("=" * 72)

    full_cos, _ = compare_tensor(
        "block6 full tokens",
        full_tokens,
        golden["block6_full_tokens"],
    )

    raw_cos, _ = compare_tensor(
        "block6 raw patch features",
        raw_patches,
        golden[
            "block6_patch_feature_raw"
        ],
    )

    current_fp16 = (
        legacy_fp16
        .detach()
        .cpu()
        .numpy()
    )

    golden_fp16 = golden[
        "block6_patch_feature_legacy_fp16"
    ]

    fp16_exact = bool(
        np.array_equal(
            current_fp16,
            golden_fp16,
        )
    )

    print()
    print(
        "legacy FP16 exact match :",
        fp16_exact
    )

    norm_cos, _ = compare_tensor(
        "normalized features",
        normalized,
        golden["normalized_feature"],
    )

    score_cos, score_max = (
        compare_tensor(
            "patch scores",
            patch_scores[0],
            golden["patch_scores"],
        )
    )

    current_idx = (
        top10_indices[0]
        .detach()
        .cpu()
        .numpy()
    )

    golden_idx = golden[
        "top10_indices"
    ]

    top10_exact = bool(
        np.array_equal(
            current_idx,
            golden_idx,
        )
    )

    overlap = len(
        set(current_idx.tolist())
        & set(golden_idx.tolist())
    )

    current_logit = float(
        image_logit.item()
    )

    golden_logit = float(
        golden["image_logit"]
    )

    current_prob = float(
        probability.item()
    )

    golden_prob = float(
        golden["fake_probability"]
    )

    current_pred = int(
        prediction.item()
    )

    golden_pred = int(
        golden["prediction"]
    )

    print()
    print(
        "Top10 current :",
        current_idx.tolist()
    )

    print(
        "Top10 golden  :",
        golden_idx.tolist()
    )

    print(
        "Top10 exact   :",
        top10_exact
    )

    print(
        "Top10 overlap :",
        f"{overlap}/10"
    )

    print()
    print(
        "Logit current :",
        f"{current_logit:.10f}"
    )

    print(
        "Logit golden  :",
        f"{golden_logit:.10f}"
    )

    print(
        "Logit abs err :",
        f"{abs(current_logit-golden_logit):.10e}"
    )

    print(
        "Prob current  :",
        f"{current_prob:.10f}"
    )

    print(
        "Prob golden   :",
        f"{golden_prob:.10f}"
    )

    print(
        "Prediction    :",
        current_pred,
        "golden:",
        golden_pred
    )

    passed = (
        full_cos > 0.999999
        and raw_cos > 0.999999
        and norm_cos > 0.999999
        and score_cos > 0.999999
        and score_max < 2e-3
        and top10_exact
        and current_pred == golden_pred
    )

    print()
    print(
        "RESULT:",
        "PASS" if passed else "CHECK"
    )

    return passed


def main():

    print("=" * 72)
    print("EVIRoute EARLY-7 DEPLOYMENT WRAPPER")
    print("=" * 72)

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "timm   :",
        timm.__version__
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device :",
        device
    )

    model = EviRouteEarly7Deploy(
        CKPT
    ).to(device)

    model.eval()

    print()
    print(
        "Deploy blocks:",
        len(model.blocks)
    )

    print(
        "Output semantic:",
        "196 patch scores"
    )

    real_pass = validate_one(
        model,
        REAL_GOLDEN,
        device,
    )

    fake_pass = validate_one(
        model,
        FAKE_GOLDEN,
        device,
    )

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    if real_pass and fake_pass:
        print(
            "Early-7 deployment wrapper "
            "vs clean PyTorch golden: PASS"
        )
    else:
        print(
            "Early-7 deployment wrapper "
            "vs clean PyTorch golden: CHECK"
        )


if __name__ == "__main__":
    main()

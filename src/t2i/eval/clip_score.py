"""
t2i.eval.clip_score — CLIP-based text-image alignment score.

Computes cosine similarity between CLIP image and text embeddings.

Convention: ``100 × cosine_similarity`` (matching the original CLIPScore paper).
All results use this same scaling; do not mix with raw cosine scores.

Preferred model: ViT-L/14. If ViT-B/32 is used, it must be documented and
results must not be compared against ViT-L/14 numbers.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import open_clip  # type: ignore
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Using ViT-B-32 for mobile CPU proxy per section 8.6.1 allowance to respect working memory limits.
_DEFAULT_MODEL = "ViT-B-32"
_DEFAULT_PRETRAINED = "openai"

# Scaling factor: 100× cosine similarity (CLIPScore convention)
CLIP_SCALE = 100.0


class CLIPScorer:
    """
    Computes CLIPScore for a list of (image, prompt) pairs.

    Parameters
    ----------
    model_name:
        Open-CLIP model architecture name.
    pretrained:
        Pretrained weights identifier.
    device:
        Target device.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        pretrained: str = _DEFAULT_PRETRAINED,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        logger.info("Loading CLIP model %s (%s)…", model_name, pretrained)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.model_name = model_name

    def score(self, image: np.ndarray, prompt: str) -> float:
        """
        Compute CLIPScore for one (image, prompt) pair.

        Returns 100 × cosine_similarity (see CLIP_SCALE).

        Parameters
        ----------
        image:
            [H, W, 3] uint8 RGB image.
        prompt:
            Text prompt used to generate the image.

        Returns
        -------
        float
            CLIPScore in [0, 100].
        """
        pil_img = PILImage.fromarray(image)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        text_tokens = self.tokenizer([prompt]).to(self.device)  # [1, 77]

        with torch.inference_mode():
            img_features = self.model.encode_image(img_tensor)  # [1, D]
            txt_features = self.model.encode_text(text_tokens)  # [1, D]

        # Normalise to unit length
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

        # Cosine similarity → [0, 1] range (clamp negatives to 0 per convention)
        cos_sim = (img_features * txt_features).sum(dim=-1).clamp(min=0.0)
        return float(cos_sim.item() * CLIP_SCALE)

    def score_batch(
        self,
        images: list[np.ndarray],
        prompts: list[str],
    ) -> dict:
        """
        Compute CLIPScore statistics over a list of (image, prompt) pairs.

        Parameters
        ----------
        images:
            List of [H, W, 3] uint8 RGB images.
        prompts:
            Corresponding text prompts.

        Returns
        -------
        dict
            ``{"mean": float, "std": float, "scores": list[float], "model": str}``
        """
        assert len(images) == len(prompts), "images and prompts must have the same length"
        scores = [self.score(img, p) for img, p in zip(images, prompts)]
        arr = np.array(scores)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "scores": scores,
            "model": self.model_name,
            "scale": f"{CLIP_SCALE}x_cosine",
            "n": len(scores),
        }

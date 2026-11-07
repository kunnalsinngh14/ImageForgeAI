"""
t2i.pipeline.prompt_encoder — CLIP text encoding for classifier-free guidance.

Tokenises the prompt and negative prompt, runs the CLIP ViT-L/14 text encoder,
handles long-prompt truncation warnings, and concatenates embeddings for CFG.

Tensor shapes (SD v1.5 defaults):
  tokens:         [1, 77]                 (max CLIP sequence length)
  text_embeddings:[1, 77, 768]            (CLIP ViT-L/14 hidden dim)
  CFG batch:      [2, 77, 768]            (uncond || cond concatenation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)

# CLIP maximum token sequence length (including BOS and EOS tokens).
CLIP_MAX_LENGTH: int = 77


@dataclass
class EncodingResult:
    """Return value of :func:`encode_prompt`."""

    # Shape: [2, 77, 768]  (uncond first, cond second) when cfg_enabled=True
    # Shape: [1, 77, 768]  (cond only) when cfg_enabled=False
    embeddings: torch.Tensor

    # True when the prompt was silently truncated to 77 tokens.
    truncated: bool

    # Number of tokens that were dropped due to truncation (0 if not truncated).
    dropped_tokens: int


class PromptEncoder:
    """
    Wraps the CLIP tokenizer and text encoder for SD v1.5.

    Parameters
    ----------
    tokenizer:
        CLIPTokenizer loaded from the SD v1.5 checkpoint.
    text_encoder:
        CLIPTextModel loaded from the SD v1.5 checkpoint.
    device:
        Target device (e.g. ``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        device: str | torch.device = "cpu",
    ) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
    ) -> EncodingResult:
        """
        Tokenise and encode a prompt (and optionally a negative prompt) for CFG.

        Steps:
        1. Tokenise both prompts; detect truncation.
        2. Run the text encoder on each token sequence.
        3. When CFG is enabled (guidance_scale != 1.0), concatenate
           [uncond_embeddings, cond_embeddings] along dim 0 to form the CFG batch.
           When guidance_scale == 1.0 (LCM no-guidance), return cond only.

        Parameters
        ----------
        prompt:
            The user's text prompt describing the desired image.
        negative_prompt:
            Text describing what to avoid. Defaults to an empty string
            (the CLIP encoder sees an all-padding sequence).
        guidance_scale:
            CFG scale. When == 1.0, unconditional embeddings are skipped.

        Returns
        -------
        EncodingResult
            Embeddings tensor and truncation metadata.
        """
        cfg_enabled = guidance_scale != 1.0

        # --- Step 1: Tokenise and detect truncation ---
        cond_tokens, cond_truncated, cond_dropped = self._tokenize(prompt)
        if cond_truncated:
            logger.warning(
                "Prompt was truncated to %d tokens; %d token(s) were dropped. " "Prompt: %.80s…",
                CLIP_MAX_LENGTH,
                cond_dropped,
                prompt,
            )

        # --- Step 2: Encode the conditional prompt ---
        # Shape: [1, 77, 768]
        cond_embeddings = self._encode_tokens(cond_tokens)

        if not cfg_enabled:
            # LCM with guidance_scale=1.0: skip the unconditional pass entirely.
            return EncodingResult(
                embeddings=cond_embeddings,
                truncated=cond_truncated,
                dropped_tokens=cond_dropped,
            )

        # --- Step 2b: Encode the unconditional (negative) prompt ---
        uncond_tokens, _, _ = self._tokenize(negative_prompt)
        # Shape: [1, 77, 768]
        uncond_embeddings = self._encode_tokens(uncond_tokens)

        # --- Step 3: Concatenate for CFG ---
        # CFG batch shape: [2, 77, 768]
        # The U-Net processes both in a single forward pass, then we split
        # the output into uncond and cond predictions inside the denoiser.
        cfg_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

        return EncodingResult(
            embeddings=cfg_embeddings,
            truncated=cond_truncated,
            dropped_tokens=cond_dropped,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> tuple[torch.Tensor, bool, int]:
        """
        Tokenise *text* and return (token_ids, truncated_flag, dropped_count).

        First tokenises without truncation to count the real length,
        then re-tokenises with truncation to get the input ids.

        Returns
        -------
        token_ids : torch.Tensor
            Shape [1, 77] on ``self.device``.
        truncated : bool
            Whether the sequence was truncated.
        dropped : int
            Number of dropped tokens (0 if not truncated).
        """
        # Tokenise without truncation to detect the true length.
        full = self.tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        # input_ids includes BOS and EOS; subtract 2 for content tokens.
        full_length: int = full.input_ids.shape[1]  # includes BOS+EOS
        truncated = full_length > CLIP_MAX_LENGTH
        dropped = max(0, full_length - CLIP_MAX_LENGTH)

        # Tokenise with truncation and padding to exactly CLIP_MAX_LENGTH.
        encoded = self.tokenizer(
            text,
            max_length=CLIP_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Shape: [1, 77]
        token_ids: torch.Tensor = encoded.input_ids.to(self.device)
        return token_ids, truncated, dropped

    def _encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the CLIP text encoder.

        Parameters
        ----------
        token_ids:
            Shape [1, 77] integer token ids on ``self.device``.

        Returns
        -------
        torch.Tensor
            Shape [1, 77, 768] — last hidden state from the CLIP encoder.
        """
        with torch.inference_mode():
            # ``last_hidden_state`` is what diffusers cross-attention expects.
            output = self.text_encoder(token_ids)
            # Shape: [1, 77, 768]
            return output.last_hidden_state


# ---------------------------------------------------------------------------
# Factory helper (used by generate.py)
# ---------------------------------------------------------------------------


def load_prompt_encoder(
    model_id: str,
    revision: str,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> PromptEncoder:
    """
    Load a CLIPTokenizer and CLIPTextModel from the Hub and return a PromptEncoder.

    Parameters
    ----------
    model_id:
        Hugging Face Hub model id (e.g. ``"runwayml/stable-diffusion-v1-5"``).
    revision:
        Pinned git commit hash.
    device:
        Target device.
    dtype:
        torch dtype for the text encoder weights.
    """
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        revision=revision,
    )

    from pathlib import Path
    import json
    import safetensors.torch
    from transformers import CLIPTextConfig

    p = Path(model_id) / "text_encoder"
    cfg_file = p / "config.json"
    w_fp16 = p / "model.fp16.safetensors"
    w_fp32 = p / "model.safetensors"

    if cfg_file.exists() and (w_fp16.exists() or w_fp32.exists()):
        w_file = w_fp16 if w_fp16.exists() else w_fp32
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        text_config = CLIPTextConfig.from_dict(cfg_dict)
        text_encoder = CLIPTextModel(text_config)
        sd = safetensors.torch.load_file(str(w_file), device="cpu")
        text_encoder.load_state_dict(sd, strict=False, assign=True)
        del sd
        text_encoder = text_encoder.to(device)
    else:
        text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            revision=revision,
            torch_dtype=dtype,
        ).to(device)
    text_encoder.eval()
    return PromptEncoder(tokenizer, text_encoder, device)

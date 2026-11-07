"""Integration tests for the full generation pipeline.

These tests use a mock denoiser that returns deterministic tensors so they
can run in CI without downloading the full SD v1.5 weights.

Run with: pytest tests/integration --mock
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Mock pipeline components
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal CLIP tokenizer mock that simulates truncation behaviour.

    Each whitespace-separated word maps to one token (plus BOS and EOS),
    so a prompt with more than 75 words is genuinely truncated to 77 tokens.
    """

    model_max_length = 77

    def __call__(self, text, max_length=77, padding=None, truncation=False, return_tensors=None):
        words = text.split()
        # BOS (0) + one token per word + EOS (0) — CLIP uses 49406/49407 in reality.
        ids = [49406] + [1000 + i for i in range(len(words))] + [49407]

        if truncation:
            ids = ids[:max_length]
            if len(ids) < max_length and padding == "max_length":
                ids = ids + [0] * (max_length - len(ids))
        elif padding == "max_length":
            ids = ids[:max_length]
            if len(ids) < max_length:
                ids = ids + [0] * (max_length - len(ids))

        input_ids = torch.tensor([ids], dtype=torch.long)

        class Output:
            pass
        out = Output()
        out.input_ids = input_ids
        return out


class MockTextEncoder(torch.nn.Module):
    def forward(self, input_ids):
        # Return dummy [1, 77, 768] embeddings
        class Output:
            last_hidden_state = torch.zeros(1, 77, 768)
        return Output()


class MockUNet(torch.nn.Module):
    """Returns zero noise predictions — deterministic for testing."""
    dtype = torch.float32

    def forward(self, latent, t, encoder_hidden_states=None, return_dict=True):
        return (torch.zeros_like(latent),)


class MockVAE(torch.nn.Module):
    """Returns a fixed pink image."""
    dtype = torch.float32

    def decode(self, latents, return_dict=True):
        h, w = latents.shape[2] * 8, latents.shape[3] * 8
        # Return [1, 3, H, W] tensor in [-1, 1]
        img = torch.zeros(1, 3, h, w)
        img[0, 0, :, :] = 0.5   # R channel
        return (img,)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.mock
def test_prompt_encoder_truncation_flag():
    """Tokeniser must detect when a prompt exceeds 77 tokens and set the flag."""
    from t2i.pipeline.prompt_encoder import PromptEncoder, CLIP_MAX_LENGTH

    tokenizer = MockTokenizer()
    text_encoder = MockTextEncoder().eval()
    encoder = PromptEncoder(tokenizer, text_encoder, device="cpu")

    # A short prompt: no truncation expected.
    short = encoder.encode_prompt(prompt="a red car", negative_prompt="", guidance_scale=1.0)
    assert short.truncated is False
    assert short.dropped_tokens == 0

    # A long prompt (> 75 content words → > 77 tokens with BOS+EOS).
    long_prompt = " ".join(f"word{i}" for i in range(100))
    long_result = encoder.encode_prompt(prompt=long_prompt, negative_prompt="", guidance_scale=1.0)
    assert long_result.truncated is True
    assert long_result.dropped_tokens > 0
    # Output shape must still be the fixed CLIP shape.
    assert long_result.embeddings.shape == (1, CLIP_MAX_LENGTH, 768)


@pytest.mark.mock
def test_prompt_encoder_cfg_shape():
    """CFG embedding must have batch size 2 when guidance_scale != 1.0."""
    from t2i.pipeline.prompt_encoder import PromptEncoder

    tokenizer = MockTokenizer()
    text_encoder = MockTextEncoder().eval()
    encoder = PromptEncoder(tokenizer, text_encoder, device="cpu")

    result = encoder.encode_prompt(
        prompt="a red car",
        negative_prompt="",
        guidance_scale=7.5,
    )
    # CFG: uncond + cond → batch=2
    assert result.embeddings.shape == (2, 77, 768)


@pytest.mark.mock
def test_decoder_output_shape():
    """VAE decoder must return [H, W, 3] uint8 image."""
    from t2i.pipeline.decoder import Decoder

    vae = MockVAE().eval()
    decoder = Decoder(vae, device="cpu", enable_tiling=False)

    latents = torch.zeros(1, 4, 32, 32)  # 32×32 latent → 256×256 image
    image = decoder.decode(latents)
    assert image.shape == (256, 256, 3)
    assert image.dtype == np.uint8


@pytest.mark.mock
def test_denoiser_step_count():
    """Denoiser must execute exactly the configured number of steps."""
    from t2i.pipeline.denoiser import Denoiser
    from diffusers import DDIMScheduler

    unet = MockUNet().eval()
    scheduler = DDIMScheduler()
    denoiser = Denoiser(unet, "ddim", num_inference_steps=5, device="cpu")

    latents = torch.zeros(1, 4, 8, 8)
    embeddings = torch.zeros(2, 77, 768)

    result = denoiser.denoise(latents, embeddings, guidance_scale=7.5, seed=42)
    assert result.steps_executed == 5
    assert result.early_stopped is False


@pytest.mark.mock
def test_denoiser_early_stop():
    """Step callback returning True must stop the denoiser early."""
    from t2i.pipeline.denoiser import Denoiser

    unet = MockUNet().eval()
    denoiser = Denoiser(unet, "ddim", num_inference_steps=10, device="cpu")

    latents = torch.zeros(1, 4, 8, 8)
    embeddings = torch.zeros(2, 77, 768)

    def stop_at_step_2(step_idx, latents):
        return step_idx >= 2  # stop after step index 2 (3rd step)

    result = denoiser.denoise(latents, embeddings, guidance_scale=7.5, seed=42,
                               step_callback=stop_at_step_2)
    assert result.early_stopped is True
    assert result.steps_executed == 3  # steps 0, 1, 2


@pytest.mark.mock
def test_config_hash_deterministic():
    """Config hash must be the same for identical configs."""
    from t2i.config import load_config
    from pathlib import Path

    cfg_path = Path("configs/baseline.yaml")
    if not cfg_path.exists():
        pytest.skip("configs/baseline.yaml not found")

    cfg1 = load_config(cfg_path)
    cfg2 = load_config(cfg_path)
    assert cfg1.config_hash() == cfg2.config_hash()


@pytest.mark.mock
def test_config_unknown_key_raises():
    """Config with unknown keys must raise a ValidationError."""
    from pydantic import ValidationError
    from t2i.config import T2IConfig, ModelConfig

    with pytest.raises(ValidationError):
        T2IConfig(
            name="bad",
            model=ModelConfig(base="x", revision="y"),
            precision="fp32",
            scheduler="ddim",
            steps=10,
            guidance_scale=7.5,
            resolution=[512, 512],
            variant="full_unet",
            lcm_lora=False,
            quantization="none",
            attention="default",
            vae_tiling=False,
            classical_postprocess=False,
            seed_base=42,
            unknown_field="this should fail",  # type: ignore
        )

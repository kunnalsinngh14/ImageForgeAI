"""
t2i.optimize.export_onnx — ONNX export and numerical validation.

Exports the U-Net, VAE decoder, and optionally the text encoder to ONNX with
fixed input shapes, then validates the exported graphs numerically against
PyTorch by running the same inputs through both and reporting max/mean
absolute error on outputs.

The exported models are required for the mobile deployment path (Section 12).

Opset: 17 (supports modern operators including SDPA when available).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel

logger = logging.getLogger(__name__)

OPSET_VERSION = 18


# ---------------------------------------------------------------------------
# U-Net export
# ---------------------------------------------------------------------------


def export_unet_to_onnx(
    unet: UNet2DConditionModel,
    output_path: Path,
    batch_size: int = 1,
    use_cfg: bool = False,
    latent_channels: int = 4,
    latent_h: int = 64,
    latent_w: int = 64,
    seq_len: int = 77,
    hidden_dim: int = 768,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Path:
    """
    Export the U-Net to ONNX with fixed input shapes.

    Input names and shapes:
      - ``latent_model_input``: [batch, 4, H/8, W/8]
      - ``timestep``:           [1] (int64)
      - ``encoder_hidden_states``: [batch, 77, 768]

    Output:
      - ``noise_pred``: [batch, 4, H/8, W/8]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual_batch = batch_size * 2 if use_cfg else batch_size

    dummy_latent = torch.randn(
        actual_batch, latent_channels, latent_h, latent_w, dtype=dtype, device=device
    )
    dummy_t = torch.tensor([999], dtype=torch.long, device=device)
    dummy_emb = torch.randn(actual_batch, seq_len, hidden_dim, dtype=dtype, device=device)

    logger.info("Exporting U-Net to ONNX at %s (opset %d)…", output_path, OPSET_VERSION)

    unet.eval()
    with torch.inference_mode():
        torch.onnx.export(
            unet,
            args=(dummy_latent, dummy_t, dummy_emb),
            f=str(output_path),
            input_names=["latent_model_input", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )

    logger.info("U-Net ONNX export saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# VAE decoder export
# ---------------------------------------------------------------------------


class _VAEDecoderWrapper(torch.nn.Module):
    """Thin wrapper to export only the VAE decoder (not the full VAE)."""

    def __init__(self, vae: AutoencoderKL) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # decode() returns a DecoderOutput; extract the sample.
        return self.vae.decode(latents, return_dict=False)[0]


def export_vae_decoder_to_onnx(
    vae: AutoencoderKL,
    output_path: Path,
    batch_size: int = 1,
    latent_channels: int = 4,
    latent_h: int = 64,
    latent_w: int = 64,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Path:
    """
    Export the VAE decoder to ONNX with fixed input shapes.

    Input:  ``latents`` [1, 4, H/8, W/8]
    Output: ``image``   [1, 3, H, W]

    Parameters
    ----------
    vae:
        AutoencoderKL to export.
    output_path:
        Target .onnx file path.
    Returns
    -------
    Path
        Path to the saved ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_latent = torch.randn(
        batch_size, latent_channels, latent_h, latent_w, dtype=dtype, device=device
    )
    wrapper = _VAEDecoderWrapper(vae).eval()

    logger.info("Exporting VAE decoder to ONNX at %s (opset %d)…", output_path, OPSET_VERSION)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            args=(dummy_latent,),
            f=str(output_path),
            input_names=["latents"],
            output_names=["image"],
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )

    logger.info("VAE decoder ONNX export saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Numerical validation
# ---------------------------------------------------------------------------


def validate_onnx_vs_pytorch(
    onnx_path: Path,
    pytorch_model,
    dummy_inputs: tuple,
    input_names: list[str],
    num_threads: int = 4,
) -> dict:
    """
    Run the same inputs through both the ONNX model and the PyTorch model,
    then report max and mean absolute error on the outputs.

    Parameters
    ----------
    onnx_path:
        Path to the ONNX model.
    pytorch_model:
        The corresponding PyTorch module.
    dummy_inputs:
        Tuple of torch tensors to feed to both models.
    input_names:
        List of ONNX input names, in the same order as ``dummy_inputs``.
    num_threads:
        Number of ONNX Runtime inter-op threads (reference thread count).

    Returns
    -------
    dict
        ``{"max_abs_error": float, "mean_abs_error": float, "output_shapes_match": bool}``
    """
    import onnxruntime as ort  # type: ignore

    # --- PyTorch reference ---
    if hasattr(pytorch_model, "eval"):
        pytorch_model.eval()
    with torch.inference_mode():
        pt_out = pytorch_model(*dummy_inputs)
    if hasattr(pt_out, "sample"):
        pt_out = pt_out.sample
    elif isinstance(pt_out, (list, tuple)):
        pt_out = pt_out[0]
    pt_numpy = pt_out.cpu().numpy()

    # Free PyTorch model memory before initializing ONNX Runtime
    del pytorch_model
    import gc

    gc.collect()

    # --- ONNX Runtime ---
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.inter_op_num_threads = num_threads
    sess_opts.enable_cpu_mem_arena = False
    sess = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    feed = {name: inp.cpu().numpy() for name, inp in zip(input_names, dummy_inputs)}
    ort_out = sess.run(None, feed)[0]

    # --- Comparison ---
    shapes_match = pt_numpy.shape == ort_out.shape
    if shapes_match:
        diff = np.abs(pt_numpy.astype(np.float32) - ort_out.astype(np.float32))
        max_err = float(diff.max())
        mean_err = float(diff.mean())
    else:
        max_err = float("nan")
        mean_err = float("nan")

    report = {
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "output_shapes_match": shapes_match,
        "pytorch_shape": tuple(pt_numpy.shape),
        "onnx_shape": tuple(ort_out.shape),
    }
    logger.info("ONNX vs PyTorch validation: %s", report)
    return report

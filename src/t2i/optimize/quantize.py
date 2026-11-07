"""
t2i.optimize.quantize — FP16 and INT8 weight quantization for the pipeline.

Implements two quantization strategies:
A. Dynamic INT8 quantization (``torch.ao.quantization.quantize_dynamic``)
   — quick, CPU-friendly, no calibration data required.
B. ONNX Runtime INT8 static quantization with calibration prompts
   — better quality, slightly slower to prepare; requires a calibration set.

FP16 conversion is also implemented here for completeness and can be applied
to any module independently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.ao.quantization as tq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FP16
# ---------------------------------------------------------------------------


def convert_to_fp16(module: torch.nn.Module) -> torch.nn.Module:
    """
    Convert all floating-point parameters and buffers to float16.

    Parameters
    ----------
    module:
        Any PyTorch module (U-Net, text encoder, VAE).

    Returns
    -------
    torch.nn.Module
        The same module, converted to float16 in-place.
    """
    module.half()
    logger.info("Module converted to FP16: %s", type(module).__name__)
    return module


# ---------------------------------------------------------------------------
# Dynamic INT8 (Option A)
# ---------------------------------------------------------------------------


def quantize_dynamic_int8(
    module: torch.nn.Module,
    layer_types: tuple = (torch.nn.Linear,),
) -> torch.nn.Module:
    """
    Apply ``torch.ao.quantization.quantize_dynamic`` to linear layers.

    Fast, calibration-free, CPU-only INT8 quantization.  Only the weight
    tensors of the specified layer types are quantized; activations remain FP32.

    Parameters
    ----------
    module:
        PyTorch module to quantize (typically the U-Net or text encoder).
    layer_types:
        Layer classes to target.  Defaults to ``(torch.nn.Linear,)``.

    Returns
    -------
    torch.nn.Module
        Quantized module.
    """
    logger.info("Applying dynamic INT8 quantization to %s…", type(module).__name__)
    quantized = tq.quantize_dynamic(
        module,
        qconfig_spec=set(layer_types),
        dtype=torch.qint8,
        inplace=False,
    )
    logger.info("Dynamic INT8 quantization complete.")
    return quantized


# ---------------------------------------------------------------------------
# ONNX Runtime static INT8 (Option B) — called after ONNX export
# ---------------------------------------------------------------------------


def quantize_onnx_int8(
    onnx_path: Path,
    output_path: Optional[Path] = None,
    calibration_data_reader=None,
) -> Path:
    """
    Apply ONNX Runtime static INT8 quantization to an exported ONNX model.

    Uses ``onnxruntime.quantization.quantize_static`` with
    ``QuantType.QInt8`` for weights and activations.

    A ``calibration_data_reader`` is required for static quantization because
    activation ranges must be measured on representative inputs.

    Parameters
    ----------
    onnx_path:
        Path to the FP32 or FP16 ONNX model.
    output_path:
        Output path for the INT8 ONNX model.  Defaults to
        ``<onnx_path.stem>_int8.onnx`` in the same directory.
    calibration_data_reader:
        An ``onnxruntime.quantization.CalibrationDataReader`` instance
        pre-loaded with calibration prompts.  If None, only weight
        quantization (dynamic) is applied.

    Returns
    -------
    Path
        Path to the saved INT8 ONNX model.
    """
    try:
        from onnxruntime.quantization import (  # type: ignore
            QuantType,
            quantize_dynamic as ort_quantize_dynamic,
            quantize_static,
        )
    except ImportError as exc:
        raise ImportError("onnxruntime-tools is required for ONNX INT8 quantization.") from exc

    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_int8.onnx"

    if calibration_data_reader is not None:
        logger.info("Applying ONNX static INT8 quantization with calibration data…")
        quantize_static(
            model_input=str(onnx_path),
            model_output=str(output_path),
            calibration_data_reader=calibration_data_reader,
            quant_type=QuantType.QInt8,
        )
    else:
        logger.info("Applying ONNX dynamic INT8 quantization (no calibration data)…")
        ort_quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )

    logger.info("ONNX INT8 model saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Model-size reporting
# ---------------------------------------------------------------------------


def model_size_mb(module: torch.nn.Module) -> float:
    """
    Estimate the in-memory parameter size of a PyTorch module in MiB.

    Parameters
    ----------
    module:
        Any PyTorch module.

    Returns
    -------
    float
        Parameter size in MiB.
    """
    total_bytes = sum(p.nelement() * p.element_size() for p in module.parameters())
    total_bytes += sum(b.nelement() * b.element_size() for b in module.buffers())
    return total_bytes / (1024**2)

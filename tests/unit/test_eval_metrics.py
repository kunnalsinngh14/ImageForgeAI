"""Tests for evaluation modules (size, memory, clip_score, etc.)."""

from t2i.eval.size import measure_component_sizes, onnx_export_size_mb
from t2i.eval.memory import RSSPoller, measure_peak_cpu_memory_mib


def test_measure_component_sizes_empty(tmp_path):
    res = measure_component_sizes(tmp_path)
    assert "components" in res
    assert "total_mb" in res
    assert res["total_mb"] == 0.0


def test_measure_component_sizes_with_dummy_weight(tmp_path):
    unet_dir = tmp_path / "unet"
    unet_dir.mkdir()
    dummy_weight = unet_dir / "diffusion_pytorch_model.safetensors"
    # Write 1 MB
    dummy_weight.write_bytes(b"\x00" * (1024 * 1024))

    res = measure_component_sizes(tmp_path)
    assert "unet" in res["components"]
    assert res["components"]["unet"] >= 1.0
    assert res["total_mb"] >= 1.0


def test_onnx_export_size(tmp_path):
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"\x00" * (512 * 1024))
    res = onnx_export_size_mb(tmp_path)
    assert "model.onnx" in res["files"]
    assert res["total_mb"] >= 0.5


def test_rss_poller():
    with RSSPoller(interval_ms=10) as poller:
        # Allocate some small memory
        _ = [0] * 100000
    assert poller.peak_mib > 0.0


def test_measure_peak_cpu_memory():
    def dummy_fn():
        return 42

    peak, ret = measure_peak_cpu_memory_mib(dummy_fn, interval_ms=10)
    assert ret == 42
    assert peak > 0.0

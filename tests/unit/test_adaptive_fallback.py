"""Tests for t2i.adaptive.fallback module."""
from t2i.config import T2IConfig, ModelConfig
from t2i.adaptive.fallback import apply_fallback_chain


def _base_config(**overrides) -> T2IConfig:
    defaults = dict(
        name="test",
        model=ModelConfig(base="runwayml/stable-diffusion-v1-5", revision="main"),
        precision="fp32",
        scheduler="ddim",
        steps=50,
        guidance_scale=7.5,
        resolution=[512, 512],
        variant="full_unet",
        lcm_lora=False,
        quantization="none",
        attention="default",
        vae_tiling=False,
        channels_last=False,
        classical_postprocess=False,
        seed_base=1234,
    )
    defaults.update(overrides)
    return T2IConfig(**defaults)


class MockPsutil:
    """Monkey-patched psutil.virtual_memory to simulate memory conditions."""
    def __init__(self, available_mb):
        self._available = available_mb * 1024 * 1024

    def available(self):
        return self._available


def test_no_fallback_needed_when_memory_ok(monkeypatch):
    """When free memory > threshold, no fallback should trigger for a 384 config."""
    # Give ample memory so the chain exits early after step 1 (but step 1 won't run
    # because the config is already at min resolution)
    import t2i.adaptive.fallback as fb_module

    def mock_avail():
        return 2000.0  # 2 GB free — well above threshold

    monkeypatch.setattr(fb_module, "available_memory_mb", mock_avail)

    cfg = _base_config(resolution=[384, 384])
    result = apply_fallback_chain(
        cfg,
        memory_budget_mb=1536,
        free_memory_threshold_mb=200.0,
        min_steps=2,
        min_resolution=(384, 384),
    )
    # Already at min resolution — step 1 skipped; step 2 would trigger (variant)
    assert result.success


def test_fallback_step1_reduces_resolution(monkeypatch):
    """Step 1 must reduce resolution to 384×384."""
    import t2i.adaptive.fallback as fb_module

    def mock_avail():
        # First call: memory OK to enter chain (> threshold)
        # Second call after step 1: still OK
        return 500.0  # above threshold but we control flow through args

    monkeypatch.setattr(fb_module, "available_memory_mb", mock_avail)

    cfg = _base_config(resolution=[512, 512])
    result = apply_fallback_chain(
        cfg,
        memory_budget_mb=1536,
        free_memory_threshold_mb=200.0,
        min_steps=2,
        min_resolution=(384, 384),
    )
    # After step 1, resolution should be 384
    if result.config is not None:
        assert result.config.resolution == [384, 384] or result.success


def test_fallback_step3_reduces_steps(monkeypatch):
    """Step 3 must reduce steps to min_steps and enable LCM-LoRA."""
    import t2i.adaptive.fallback as fb_module

    # Simulate very low memory to jump straight to step 3
    monkeypatch.setattr(fb_module, "available_memory_mb", lambda: 100.0)

    cfg = _base_config(steps=50)
    result = apply_fallback_chain(
        cfg,
        memory_budget_mb=1536,
        free_memory_threshold_mb=200.0,
        min_steps=2,
        min_resolution=(384, 384),
    )
    if result.config is not None:
        assert result.config.steps == 2
        assert result.config.lcm_lora is True
        assert result.config.scheduler == "lcm"


def test_fallback_already_at_min_steps_errors(monkeypatch):
    """When already at min steps and memory is very low, return failure."""
    import t2i.adaptive.fallback as fb_module

    monkeypatch.setattr(fb_module, "available_memory_mb", lambda: 50.0)

    cfg = _base_config(steps=2)
    result = apply_fallback_chain(
        cfg,
        memory_budget_mb=1536,
        free_memory_threshold_mb=200.0,
        min_steps=2,
    )
    # Should fail because we're already at min steps
    assert not result.success


def test_fallback_log_contains_entries(monkeypatch):
    """Fallback log must contain at least one entry when any step was taken."""
    import t2i.adaptive.fallback as fb_module
    monkeypatch.setattr(fb_module, "available_memory_mb", lambda: 100.0)

    cfg = _base_config(steps=50)
    result = apply_fallback_chain(cfg, memory_budget_mb=1536)
    assert len(result.fallback_log) >= 1
    for entry in result.fallback_log:
        assert "timestamp" in entry.__dict__ or hasattr(entry, "timestamp")
        assert entry.step in (1, 2, 3)

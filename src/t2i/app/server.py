"""
t2i.app.server — Local Web UI and API for image generation.

Adheres strictly to Section 8.7:
- Inputs: prompt, negative prompt, seed, resolution, step count, mode (baseline,
          optimized, adaptive), latency budget, memory budget, quality preference,
          classical post-processing toggle.
- Outputs: generated image, before/after post-processing comparison when enabled,
           quality report, latency, peak memory, fallback log (if any).
- 100% local, no hosted external API calls.

Performance notes (CPU-only machine, ~1 GB free RAM):
- Generator instances are cached by config hash — weights load once, then reused.
- Generation runs in a daemon background thread; the Gradio callback is a Python
  generator that *yields* partial UI states between stages so the browser is never
  frozen.
- torch.set_num_threads(4) is set at import time to match the mobile-proxy profile.
- Default resolution is 256×256 (4× fewer latent activations than 512×512).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Generator as PyGenerator

import gradio as gr
import torch
from PIL import Image

from t2i.adaptive.budget import load_profiles, select_config
from t2i.adaptive.fallback import apply_fallback_chain
from t2i.config import T2IConfig, load_config
from t2i.pipeline.generate import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CPU thread cap — match mobile proxy (4 cores)
# ---------------------------------------------------------------------------
torch.set_num_threads(4)
logger.info("torch CPU threads set to 4")

# ---------------------------------------------------------------------------
# Module-level Generator cache (singleton per config hash)
# ---------------------------------------------------------------------------
_generator_lock = threading.Lock()
_generator_cache: dict[str, Generator] = {}


def _get_or_create_generator(cfg: T2IConfig) -> Generator:
    """Return a cached Generator or create a new one, evicting the old one first."""
    key = cfg.config_hash()
    if key not in _generator_cache:
        if _generator_cache:
            old_key = next(iter(_generator_cache))
            old_gen = _generator_cache.pop(old_key)
            del old_gen
            gc.collect()
            logger.info("Evicted cached Generator (config=%s)", old_key)
        logger.info("Creating new Generator for config=%s", key)
        _generator_cache[key] = Generator(cfg, results_dir=Path("results"), device=None)
    return _generator_cache[key]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_config(
    mode: str,
    steps: int,
    res_w: int,
    res_h: int,
    seed: int,
    classical_pp: bool,
    latency_budget_s: float,
    memory_budget_mb: float,
    quality_pref: str,
    guidance_scale: float = 7.5,
    variant: str = "dreamshaper_v8",
) -> tuple[T2IConfig, list[str]]:
    """Resolve UI parameters to a T2IConfig. Returns (cfg, fallback_notes)."""
    fallback_notes: list[str] = []

    if mode == "baseline":
        cfg = load_config("configs/baseline.yaml").model_copy(
            update={
                "steps": int(steps),
                "resolution": [res_w, res_h],
                "classical_postprocess": classical_pp,
                "seed_base": int(seed),
                "guidance_scale": float(guidance_scale),
                "variant": variant,
            }
        )
    elif mode == "optimized":
        cfg = load_config("configs/optimized.yaml").model_copy(
            update={
                "steps": int(steps),
                "resolution": [res_w, res_h],
                "classical_postprocess": classical_pp,
                "seed_base": int(seed),
                "guidance_scale": float(guidance_scale),
                "variant": variant,
            }
        )
    elif mode == "adaptive":
        profiles_path = Path("configs/adaptive_profiles.yaml")
        if profiles_path.exists():
            profiles = load_profiles(str(profiles_path))
            budget_res = select_config(
                profiles,
                latency_budget_s=latency_budget_s,
                memory_budget_mb=memory_budget_mb,
                quality_preference=quality_pref.lower(),  # type: ignore[arg-type]
            )
            if budget_res.selected is not None:
                sel = budget_res.selected
                cfg = load_config("configs/optimized.yaml").model_copy(
                    update={
                        "steps": sel.steps,
                        "resolution": sel.resolution,
                        "precision": sel.precision,
                        "classical_postprocess": sel.classical_postprocess,
                        "seed_base": int(seed),
                    }
                )
                fallback_notes.append(f"Budget controller: {budget_res.reason}")
            else:
                fb_res = apply_fallback_chain(
                    load_config("configs/optimized.yaml"),
                    memory_budget_mb=memory_budget_mb,
                    latency_budget_s=latency_budget_s,
                )
                if fb_res.config is not None:
                    cfg = fb_res.config
                    fallback_notes.append(
                        "Fallback chain triggered: "
                        + " -> ".join(e.action for e in fb_res.fallback_log)
                    )
                else:
                    raise RuntimeError(fb_res.error_reason or "Adaptive fallback failed")
        else:
            cfg = load_config("configs/optimized.yaml")
            fallback_notes.append("Profiles not found — defaulted to optimized config.")
    else:
        cfg = load_config("configs/baseline.yaml")

    return cfg, fallback_notes


# ---------------------------------------------------------------------------
# Gradio generator callback (streams partial updates to the browser)
# ---------------------------------------------------------------------------

# Sentinel values used through the status queue
_DONE = object()
_ERROR = object()


def generate_image_ui(
    prompt: str,
    negative_prompt: str,
    seed: int,
    resolution_str: str,
    steps: int,
    mode: str,
    latency_budget_s: float,
    memory_budget_mb: float,
    quality_pref: str,
    classical_pp: bool,
    guidance_scale: float = 7.5,
    variant: str = "dreamshaper_v8",
) -> PyGenerator:
    """Gradio generator callback — yields (img, img_before, metrics, qr, fb, status)
    partial updates so the browser shows live progress during long CPU inference.

    The heavy generation runs in a daemon thread; this generator polls a queue
    and yields status strings every 0.5 s so the UI stays responsive.
    """
    # --- Parse resolution ---
    try:
        res_w, res_h = map(int, resolution_str.split("x"))
    except Exception:
        res_w, res_h = 256, 256

    # Yield an immediate "starting" state so the button spinner appears instantly
    yield (
        None,
        None,
        "",
        "",
        "",
        f"⏳ Starting generation ({mode} mode, {res_w}×{res_h}, {steps} step{'s' if steps != 1 else ''})…",
    )

    # --- Build config ---
    try:
        cfg, fallback_notes = _build_config(
            mode=mode,
            steps=int(steps),
            res_w=res_w,
            res_h=res_h,
            seed=int(seed),
            classical_pp=classical_pp,
            latency_budget_s=latency_budget_s,
            memory_budget_mb=memory_budget_mb,
            quality_pref=quality_pref,
            guidance_scale=float(guidance_scale),
            variant=variant,
        )
    except Exception as exc:
        yield None, None, "", "", "", f"❌ Config error: {exc}"
        return

    num_steps = int(cfg.steps)
    est_s = num_steps * 15 + 20  # rough estimate: 15s/step + 20s overhead on constrained RAM

    yield (
        None,
        None,
        "",
        "",
        "\n".join(fallback_notes) if fallback_notes else "None",
        f"⚙️ Config ready. Estimated ~{est_s}s. Loading model (once only)…",
    )

    # --- Thread-based generation ---
    # We put status messages into a queue from the worker thread;
    # the main thread polls and yields them so Gradio can stream updates.
    status_q: queue.Queue = queue.Queue()
    result_holder: list = []  # [GenerationResult] or [Exception]

    def _worker():
        try:
            with _generator_lock:
                status_q.put(("status", "🔤 Encoding prompt (CLIP text encoder)…"))
                gen = _get_or_create_generator(cfg)

                step_times_seen: list[float] = []
                step_start = [time.perf_counter()]

                def _step_cb(step_idx: int, latents) -> bool:  # noqa: ANN001
                    elapsed = time.perf_counter() - step_start[0]
                    step_times_seen.append(elapsed)
                    avg = sum(step_times_seen) / len(step_times_seen)
                    remaining = avg * (num_steps - step_idx - 1)
                    step_start[0] = time.perf_counter()
                    pct = int(100 * (step_idx + 1) / num_steps)
                    status_q.put(
                        (
                            "status",
                            f"🔁 Denoising step {step_idx + 1}/{num_steps} ({pct}%) "
                            f"— step took {elapsed:.1f}s, ~{remaining:.0f}s remaining…",
                        )
                    )
                    return False  # do not early-stop

                result = gen.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=int(seed),
                    step_callback=_step_cb,
                )
            result_holder.append(result)
            status_q.put((_DONE, result))
        except Exception as exc:
            logger.exception("Generation failed")
            result_holder.append(exc)
            status_q.put((_ERROR, exc))

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()

    # --- Poll the queue and stream updates ---
    while True:
        try:
            msg_type, payload = status_q.get(timeout=0.5)
        except queue.Empty:
            if not worker_thread.is_alive() and status_q.empty():
                yield None, None, "", "", "", "❌ Generation thread exited unexpectedly."
                return
            # Nothing new yet — re-yield the last status (keeps spinner alive)
            yield (
                None,
                None,
                "",
                "",
                "\n".join(fallback_notes) if fallback_notes else "None",
                "⏳ Working… (image generating — please wait)",
            )
            continue

        if msg_type == "status":
            yield (
                None,
                None,
                "",
                "",
                "\n".join(fallback_notes) if fallback_notes else "None",
                payload,
            )
            continue

        if msg_type is _ERROR:
            err = payload
            yield None, None, "", "", "", f"❌ Generation error: {err}"
            return

        if msg_type is _DONE:
            result = payload
            break

    # --- Format final outputs ---
    total_time = (
        result.time_encode_s
        + result.time_denoise_s
        + result.time_decode_s
        + result.time_postprocess_s
    )
    metrics_text = (
        f"**✅ Done in {total_time:.1f}s**\n\n"
        f"| Stage | Time |\n|---|---|\n"
        f"| Encode (CLIP) | {result.time_encode_s:.2f}s |\n"
        f"| Denoise ({result.steps_executed} steps) | {result.time_denoise_s:.2f}s |\n"
        f"| Decode (VAE) | {result.time_decode_s:.2f}s |\n"
        f"| Post-process (CV) | {result.time_postprocess_s:.2f}s |\n\n"
        f"**Peak Memory:** {result.peak_memory_mib:.1f} MiB ({result.memory_type})  \n"
        f"**Prompt truncated:** {result.prompt_truncated} "
        f"({result.prompt_dropped_tokens} tokens dropped)"
    )
    qr_text = json.dumps(result.quality_report or {}, indent=2)
    fb_text = "\n".join(fallback_notes) if fallback_notes else "None"
    img_final = Image.fromarray(result.image)
    img_before = (
        Image.fromarray(result.image_before_postprocess)
        if result.image_before_postprocess is not None
        else None
    )
    yield (
        img_final,
        img_before,
        metrics_text,
        qr_text,
        fb_text,
        f"✅ Generated in {total_time:.1f}s",
    )


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Build Gradio UI interface."""
    with gr.Blocks(title="ImageForgeAI — Resource-Efficient Text-to-Image Generation") as demo:
        gr.Markdown("# 🚀 ImageForgeAI: Resource-Efficient Text-to-Image Generation")
        device_label = (
            "NVIDIA GeForce GTX 1050 (4 GB CUDA)"
            if torch.cuda.is_available()
            else "CPU (4 threads)"
        )
        gr.Markdown(
            f"**100% local** — Stable Diffusion v1.5 (DreamShaper v8 & Full U-Net) running on **{device_label}**.\n\n"
            "> ⚡ **Hardware Acceleration Enabled:** Photorealistic generation with DreamShaper v8, DPM-Solver++ & CFG on your dedicated GPU."
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    value="a vibrant red apple resting on a rustic dark wooden table, studio lighting, highly detailed",
                    lines=2,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="lowres, bad anatomy, bad quality, blurry, oversaturated, deformed, grainy",
                    lines=2,
                )
                mode = gr.Radio(
                    label="Pipeline Mode",
                    choices=["baseline", "optimized", "adaptive"],
                    value="optimized",
                )

                with gr.Accordion("Generation Settings", open=True):
                    resolution = gr.Dropdown(
                        label="Resolution (512×512 recommended for high fidelity)",
                        choices=["256x256", "384x384", "512x512"],
                        value="512x512",
                    )
                    steps = gr.Slider(
                        label="Steps (15–20 recommended for DPM-Solver++)",
                        minimum=1,
                        maximum=30,
                        value=20,
                        step=1,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (CFG - Prompt Adherence)",
                        minimum=1.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                    )
                    model_variant = gr.Dropdown(
                        label="Model Variant",
                        choices=["dreamshaper_v8", "full_unet", "block_pruned"],
                        value="dreamshaper_v8",
                    )
                    seed = gr.Number(label="Seed", value=1234, precision=0)
                    classical_pp = gr.Checkbox(
                        label="Enable Classical CV Post-Processing", value=True
                    )

                with gr.Accordion("Adaptive Budget Settings", open=False):
                    latency_budget = gr.Slider(
                        label="Latency Budget (s)",
                        minimum=10.0,
                        maximum=300.0,
                        value=120.0,
                        step=5.0,
                    )
                    memory_budget = gr.Slider(
                        label="Memory Budget (MB)",
                        minimum=500,
                        maximum=4000,
                        value=900,
                        step=64,
                    )
                    quality_pref = gr.Dropdown(
                        label="Quality Preference",
                        choices=["fast", "balanced", "quality"],
                        value="fast",
                    )

                generate_btn = gr.Button("🎨 Generate Image", variant="primary")

            with gr.Column(scale=1):
                # Live status box — always visible
                status_box = gr.Textbox(
                    label="⚡ Live Status",
                    value="Ready — click Generate Image to start",
                    lines=2,
                    interactive=False,
                )
                image_output = gr.Image(label="Generated Image", type="pil")
                image_before_output = gr.Image(label="Before Classical Post-Processing", type="pil")
                metrics_output = gr.Markdown(label="Performance Metrics")

                with gr.Accordion("Quality Report (No-Reference CV Metrics)", open=False):
                    qr_output = gr.Code(label="Quality Metrics", language="json")

                with gr.Accordion("Adaptive / Fallback Log", open=False):
                    fb_output = gr.Textbox(label="Fallback Details", lines=3)

        generate_btn.click(
            fn=generate_image_ui,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                resolution,
                steps,
                mode,
                latency_budget,
                memory_budget,
                quality_pref,
                classical_pp,
                guidance_scale,
                model_variant,
            ],
            outputs=[
                image_output,
                image_before_output,
                metrics_output,
                qr_output,
                fb_output,
                status_box,
            ],
        )

        gr.Markdown(
            "<div style='text-align: center; margin-top: 2rem; color: #888; font-size: 0.9em;'>"
            "ImageForgeAI &bull; © 2026 Team Nexus"
            "</div>"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Launch ImageForgeAI local web application."""
    p = argparse.ArgumentParser(description="Launch ImageForgeAI web application.")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

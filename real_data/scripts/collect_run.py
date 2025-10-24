import os, json, time, hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import torch
import pandas as pd
from tqdm import tqdm
from diffusers import DiffusionPipeline

# ---------- Helpers ----------

# Resolve project root (one level above /scripts)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT_ROOT = os.path.join(PROJECT_ROOT, "data", "runs")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def noise_from_scheduler(scheduler, step_idx: int, timestep: int):
    """
    Returns (noise_scalar, noise_type) using either alphas_cumprod (logSNR) or sigmas.
    """
    # Prefer logSNR (works across many schedulers)
    if hasattr(scheduler, "alphas_cumprod"):
        a2 = scheduler.alphas_cumprod.float()[int(timestep)].clamp_(1e-6, 1 - 1e-6)
        logsnr = float(torch.log(a2) - torch.log1p(-a2))
        return logsnr, "logsnr"
    # Fallback to sigma-parametrized schedulers
    if hasattr(scheduler, "sigmas"):
        sigma = float(scheduler.sigmas.float()[int(step_idx)])
        return sigma, "sigma"
    raise RuntimeError("Scheduler exposes neither alphas_cumprod nor sigmas; cannot compute noise level.")

# ---------- Data classes ----------

@dataclass
class RunMeta:
    run_id: str
    created_at: str
    model_id: str
    num_inference_steps: int
    vae_scaling_factor: float
    scheduler_class: str
    scheduler_config: Dict[str, Any]
    timesteps: List[int]
    seed: Optional[int] = None
    device_dtype: str = "float32"
    libs: Dict[str, str] = None

# ---------- Collector ----------

class RunCollector:
    def __init__(self, out_dir: str, shard_size: int = 5000):
        self.out_dir = out_dir
        self.latent_dir = os.path.join(out_dir, "latents")
        os.makedirs(self.latent_dir, exist_ok=True)
        self.shard_size = shard_size
        self._latent_buffer: List[Dict[str, Any]] = []
        self._index_rows: List[Dict[str, Any]] = []
        self._row_counter = 0
        self._shard_id = 0

    def append(self, *, run_id: str, sample_id: str, step_idx: int, timestep: int,
               xt: torch.Tensor, noise_scalar: float, noise_type: str, image_path: str):
        """
        Store one (sample, step) latent and a small index row.
        xt must be shape [4,H,W] and float32 CPU tensor.
        """
        assert xt.ndim == 3 and xt.shape[0] == 4 and xt.dtype == torch.float32, "xt must be [4,H,W] float32"
        latent_row = {
            "run_id": run_id,
            "sample_id": sample_id,
            "step_idx": int(step_idx),
            "timestep": int(timestep),
            "xt": xt,  # [4,H,W] CPU float32
        }
        self._latent_buffer.append(latent_row)

        latent_ref = f"latents/latents-{self._shard_id:05d}.pt:row={self._row_counter % self.shard_size}"
        self._index_rows.append({
            "run_id": run_id,
            "sample_id": sample_id,
            "step_idx": int(step_idx),
            "timestep": int(timestep),
            "noise_scalar": float(noise_scalar),
            "noise_type": noise_type,
            "latent_path": latent_ref,
            "image_path": image_path,
        })
        self._row_counter += 1

        if len(self._latent_buffer) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self):
        shard_path = os.path.join(self.latent_dir, f"latents-{self._shard_id:05d}.pt")
        torch.save(self._latent_buffer, shard_path)
        self._latent_buffer.clear()
        self._shard_id += 1

    def finalize(self, steps_parquet_path: str):
        if self._latent_buffer:
            self._flush_shard()
        # Write steps.parquet
        df = pd.DataFrame(self._index_rows)
        df.to_parquet(steps_parquet_path, index=False)
        return df

# ---------- Main API you’ll call from your generation script ----------

def run_generation_and_collect(
    prompt: str,
    model_id: str = "stabilityai/stable-diffusion-2-1-base",
    num_inference_steps: int = 25,
    num_images_per_prompt: int = 10,
    out_root: str = DEFAULT_OUT_ROOT,
    seed: Optional[int] = 1234,
    device: Optional[str] = None,
    
):
    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "mps" else torch.float32

    # Create run folder
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    # Pipeline
    pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    pipe.enable_attention_slicing()

    # Fix seed
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # Build minimal meta
    vae_scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    meta = RunMeta(
        run_id=run_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_id=model_id,
        num_inference_steps=num_inference_steps,
        vae_scaling_factor=float(vae_scale),
        scheduler_class=pipe.scheduler.__class__.__name__,
        scheduler_config=pipe.scheduler.config.__dict__.copy(),
        timesteps=[],
        seed=seed,
        device_dtype=str(dtype),
        libs={
            "diffusers": getattr(__import__("diffusers"), "__version__", "unknown"),
            "torch": torch.__version__,
        },
    )

    # Collector
    collector = RunCollector(out_dir=out_dir, shard_size=5000)

    # Track sample ids within this call
    # We’ll number images in this run contiguously: 000001, 000002, ...
    next_sample_base = 1

    # New-style callback (Diffusers >= 0.27):
    def on_step_end_cb(pipe_obj, step_idx: int, timestep: int, callback_kwargs: Dict[str, Any]):
        latents = callback_kwargs["latents"].detach().float().cpu()  # [B,4,H,W]
        # Record timesteps array once (first call)
        if not meta.timesteps:
            # pipe.scheduler.timesteps is a tensor like [t_max, ..., t_min]
            meta.timesteps = [int(t.item()) for t in pipe_obj.scheduler.timesteps]
        # Compute noise scalar from scheduler
        noise_scalar, noise_type = noise_from_scheduler(pipe_obj.scheduler, step_idx, int(timestep))

        B = latents.shape[0]
        for b in range(B):
            sample_id = f"{next_sample_base + b:06d}"
            # Images will be written at the end; we know the path now
            image_path = os.path.join("images", f"result_{sample_id}.png")
            collector.append(
                run_id=meta.run_id,
                sample_id=sample_id,
                step_idx=step_idx,
                timestep=int(timestep),
                xt=latents[b],  # [4,H,W]
                noise_scalar=noise_scalar,
                noise_type=noise_type,
                image_path=image_path,
            )
        # Return kwargs as required by API
        callback_kwargs["latents"] = callback_kwargs["latents"]
        return callback_kwargs

    # Save prompts.jsonl (append mode; simple one-line per sample)
    prompts_path = os.path.join(out_dir, "prompts.jsonl")
    with open(prompts_path, "a", encoding="utf-8") as f:
        for i in range(num_images_per_prompt):
            sid = f"{next_sample_base + i:06d}"
            line = {"sample_id": sid, "prompt": prompt, "negative_prompt": None}
            f.write(json.dumps(line) + "\n")

    # Generate
    images = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        callback_on_step_end=on_step_end_cb,
    ).images

    # Write final images
    for i, img in enumerate(images):
        sample_id = f"{next_sample_base + i:06d}"
        path = os.path.join(out_dir, "images", f"result_{sample_id}.png")
        img.save(path)

    # Finalize index + meta
    steps_path = os.path.join(out_dir, "steps.parquet")
    df = collector.finalize(steps_parquet_path=steps_path)

    # Add hashes for quick integrity checks (optional but useful)
    try:
        steps_bytes = pd.read_parquet(steps_path).to_csv(index=False).encode("utf-8")
        steps_hash = sha256_bytes(steps_bytes)
    except Exception:
        steps_hash = "unavailable"
    meta_dict = asdict(meta)
    meta_dict["steps_sha256_csv"] = steps_hash
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)

    print(f"Wrote run to: {out_dir}")
    print(df.head())
    return out_dir

# ---------- CLI entry (example) ----------

if __name__ == "__main__":
    run_generation_and_collect(
        prompt="A hand",
        num_inference_steps=25,
        num_images_per_prompt=1,
        seed=1234,
    )

# R1 E2 small check on Kaggle T4 x2: the same edits and seeds as experiments/r1_e1/check.py on copies of the same
# generated face, attacked through the Qwen3-VL path alone and through Qwen3-VL and the VAE together. Clean and
# VAE-only outputs are reused from the E1 check; one clean edit is repeated to see whether those outputs reproduce.
# Not a pilot: one photo cannot show a protection rate. Results go to /kaggle/working, weights to /tmp.
import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

MODEL = "Qwen/Qwen-Image-2.1-Turbo"
MODEL_REV = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"
GGUF_REPO = "unsloth/Qwen-Image-2.1-Turbo-GGUF"
GGUF_REV = "5d1f9f759ea877ae556cda88dbec958b62ad3809"
GGUF_FILE = "qwen-image-2.1-turbo-Q4_K_M.gguf"
DIFFUSERS = "86ef420b17363c28e36ebad6b58555e9aa8feda2"
TRANSFORMERS = "5.22.1"

BUDGET_SECONDS = 7200
ATTACK_BUDGET_SECONDS = 1800
SIZE = 768
EPS = 16 / 255
STEP = 0.02
ITERS = 200
PATCH, MERGE, TEMPORAL = 16, 2, 2  # processor/ of MODEL_REV
MIN_PIXELS, MAX_PIXELS = 65536, 16777216
OUT = "/kaggle/working"
START = time.time()

SUBJECT = ("Candid photograph of a fictional adult woman in her thirties wearing a grey sweater, "
           "standing on a city street, face clearly visible, natural daylight.")
EDITS = {
    "raincoat": "Change the person's upper clothing to a bright red raincoat. "
    "Keep the face, pose and background unchanged.",
    "snow": "Change the background to a snowy mountain village. Keep the person, face and clothing unchanged.",
}
SEEDS = [1, 2, 3, 4]  # the seeds with complete pairs in the E1 check


def pixel_values(x):
    # Differentiable replacement for Qwen2VLImageProcessor on an RGB tensor in [-1, 1]. The checkpoint normalizes
    # with mean and std 0.5, which maps 8-bit pixels to this same scale. Its 8-bit rounding after a resize is left out.
    height, width = smart_resize(x.shape[-2], x.shape[-1], PATCH * MERGE, MIN_PIXELS, MAX_PIXELS)
    if (height, width) != x.shape[-2:]:
        x = torch.nn.functional.interpolate(x, size=(height, width), mode="bicubic", antialias=True)
    patches, grid_h, grid_w = Qwen2VLImageProcessor.patchify(None, x, PATCH, MERGE, TEMPORAL)
    return patches[0], torch.tensor([[1, grid_h, grid_w]], device=x.device)


def vision_features(visual, x):
    # The pooled tokens and the deepstack features are what the text encoder takes from the vision model.
    patches, grid = pixel_values(x)
    output = visual(patches, grid_thw=grid, return_dict=True)
    return [output.pooler_output, *output.deepstack_features]


def vision_distance(visual, x, clean):
    return sum(torch.nn.functional.mse_loss(f, c) / c.var() for f, c in zip(vision_features(visual, x), clean))


if __name__ != "__main__":
    # experiments/r1_e2/parity.py imports the functions above; everything below needs the GPUs.
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize
    import torch
else:
    report = {"model_rev": MODEL_REV, "gguf": [GGUF_REPO, GGUF_REV, GGUF_FILE], "diffusers": DIFFUSERS,
              "budget_seconds": BUDGET_SECONDS, "size": SIZE, "eps": EPS, "step": STEP, "iters": ITERS,
              "attacks": {}, "cases": []}

    def save():
        with open(f"{OUT}/e2.json", "w") as f:
            json.dump(report, f, indent=1, default=str)

    os.environ["HF_HOME"] = "/tmp/hf"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS}",
                    f"transformers=={TRANSFORMERS}", "accelerate", "gguf"], check=True)

    import numpy as np
    import torch
    from accelerate.hooks import remove_hook_from_module
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        GGUFQuantizationConfig,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor
    from huggingface_hub import hf_hub_download, snapshot_download
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize

    report["torch"] = torch.__version__
    report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    DTYPE = torch.bfloat16

    class TwoGpuPipeline(QwenImage21Pipeline):
        # Denoising on cuda:0; the text encoder is split over cuda:1 and CPU and returns to cuda:0.
        @property
        def _execution_device(self):
            return torch.device("cuda:0")

    def dequantize_custom_params(model):
        # See experiments/r1_e0/e0.py: QwenImage21ZeroCenterRMSNorm reads its GGUF weight directly.
        for module in model.modules():
            if isinstance(module, GGUFLinear):
                continue
            for name, param in list(module.named_parameters(recurse=False)):
                if isinstance(param, GGUFParameter):
                    setattr(module, name, torch.nn.Parameter(dequantize_gguf_tensor(param), requires_grad=False))

    def free_transformer_during(vae_call, transformer):
        def wrapped(*args, **kwargs):
            transformer.to("cpu")
            torch.cuda.empty_cache()
            try:
                return vae_call(*args, **kwargs)
            finally:
                torch.cuda.empty_cache()
                transformer.to("cuda:0")
        return wrapped

    def build():
        base = snapshot_download(MODEL, revision=MODEL_REV, ignore_patterns=["transformer/*.safetensors", "assets/*"])
        gguf = hf_hub_download(GGUF_REPO, GGUF_FILE, revision=GGUF_REV)
        with open(f"{base}/model_index.json") as f:
            sigmas = json.load(f)["sample_sigmas"]
        transformer = QwenImage21Transformer2DModel.from_single_file(
            gguf, config=base, subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE), torch_dtype=DTYPE,
        )
        dequantize_custom_params(transformer)
        transformer.to("cuda:0")
        vae = AutoencoderKLQwenImage21.from_pretrained(f"{base}/vae", torch_dtype=DTYPE).to("cuda:0")
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            f"{base}/text_encoder", dtype=DTYPE, device_map="auto", max_memory={1: "13GiB", "cpu": "24GiB"},
        )
        vae.encode = free_transformer_during(vae.encode, transformer)
        vae.decode = free_transformer_during(vae.decode, transformer)
        return TwoGpuPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(f"{base}/scheduler"),
            vae=vae, text_encoder=text_encoder,
            processor=Qwen3VLProcessor.from_pretrained(f"{base}/processor"),
            transformer=transformer, sample_sigmas=sigmas,
        )

    def make_images(pipe):
        # The face is made as in r1_e1/check.py, so that its clean and VAE-only outputs stay comparable.
        source = pipe(prompt=SUBJECT, width=1024, height=1024, output_resolution=1024, use_kv_cache=True,
                      generator=torch.Generator("cpu").manual_seed(100)).images[0]
        vae, processor = pipe.vae, pipe.image_processor
        source = processor.resize(source.convert("RGBA"), width=SIZE, height=SIZE)
        source.convert("RGB").save(f"{OUT}/source.png")
        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
        encode = type(vae).encode.__get__(vae)  # vae.encode is wrapped to move the transformer on every call
        vae.requires_grad_(False)
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to("cuda:0", DTYPE)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to("cuda:0", DTYPE)
        # A copy on cuda:0: cuda:1 holds the text encoder with about 1.3 GiB free at its E0 peak.
        visual = remove_hook_from_module(copy.deepcopy(pipe.text_encoder.model.visual), recurse=True).to("cuda:0")
        visual.requires_grad_(False)

        def to_tensor(image):
            return processor.preprocess(image.convert("RGBA"), width=SIZE, height=SIZE).unsqueeze(2).to("cuda:0", DTYPE)

        def latent(x):
            return (encode(x).latent_dist.mode() - mean) / std

        def embeds(image):
            return pipe._get_qwen_prompt_embeds(EDITS["raincoat"], image=[image], device=torch.device("cuda:0"))[0]

        x = to_tensor(source)
        with torch.no_grad():
            clean_latent = latent(x)
            clean_vision = vision_features(visual, x[:, :3, 0])
            clean_embeds = embeds(source)
        # The real processor on the file an attacker receives must match the replacement, or the attack is off target.
        expected = pipe.processor.image_processor(source.convert("RGB"), return_tensors="pt")["pixel_values"]
        # In float32: BF16 alone rounds this scale by up to 0.00195, which says nothing about the replacement.
        exact = processor.preprocess(source.convert("RGBA"), width=SIZE, height=SIZE)[:, :3]
        parity = (expected - pixel_values(exact)[0].cpu()).abs().max().item()
        report["preprocess_max_difference"] = parity
        if parity > 1e-3:
            raise ValueError(f"preprocess mismatch {parity}")

        rgb = torch.tensor([1.0, 1.0, 1.0, 0.0], device="cuda:0", dtype=DTYPE).view(1, 4, 1, 1, 1)

        def attack(mode):
            torch.manual_seed(1)
            torch.cuda.reset_peak_memory_stats(0)
            adv = (x + (torch.rand_like(x.float()).to(DTYPE) * 2 - 1) * EPS * rgb).clamp(-1, 1)
            losses, t = [], time.time()
            for i in range(ITERS):
                if time.time() - t > ATTACK_BUDGET_SECONDS:
                    raise TimeoutError(f"attack {mode} stopped at iteration {i}")
                step = STEP - (STEP - STEP / 100) / ITERS * i
                adv.requires_grad_(True)
                distance = vision_distance(visual, adv[:, :3, 0], clean_vision)
                # The two losses have unrelated scales, so each gradient gets the same L1 weight before the sign.
                grad, = torch.autograd.grad(distance, [adv])
                direction = -grad / grad.abs().sum()
                entry = [round(distance.item(), 4)]
                if mode == "both":
                    norm = latent(adv).norm()
                    grad, = torch.autograd.grad(norm, [adv])
                    direction = direction + grad / grad.abs().sum()
                    entry.append(round(norm.item(), 2))
                losses.append(entry)
                adv = adv.detach() - direction.sign() * step * rgb
                adv = torch.minimum(torch.maximum(adv, x - EPS), x + EPS).clamp(-1, 1)
            # Everything below is measured on the 8-bit file.
            pixels = ((adv[0, :3, 0].float().cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).round().astype(np.uint8)
            Image.fromarray(pixels, "RGB").save(f"{OUT}/protected_{mode}.png")
            image = Image.open(f"{OUT}/protected_{mode}.png").convert("RGB")
            difference = np.asarray(image, dtype=np.float64) - np.asarray(source.convert("RGB"), dtype=np.float64)
            final = to_tensor(image)
            with torch.no_grad():
                result = {
                    "seconds": round(time.time() - t, 1),
                    "peak_gib_cuda0": round(torch.cuda.max_memory_allocated(0) / 2**30, 2),
                    "losses": losses[:3] + losses[-3:],
                    "latent_norm": [round(clean_latent.norm().item(), 2), round(latent(final).norm().item(), 2)],
                    "vision_distance": round(vision_distance(visual, final[:, :3, 0], clean_vision).item(), 4),
                    "embeds_rmse_vs_clean": round(((embeds(image) - clean_embeds).float() ** 2).mean().sqrt().item(), 4),
                    "clean_embeds_rms": round((clean_embeds.float() ** 2).mean().sqrt().item(), 4),
                    "linf_8bit": int(np.abs(difference).max()),
                    "psnr_db": round(float(10 * np.log10(255**2 / (difference**2).mean())), 2),
                    "sha256": hashlib.sha256(open(f"{OUT}/protected_{mode}.png", "rb").read()).hexdigest(),
                }
            report["attacks"][mode] = result
            print("attack", mode, result["seconds"], result["vision_distance"], result["psnr_db"], flush=True)
            save()
            return image

        images = {"source": source.convert("RGB"), "vision": attack("vision"), "both": attack("both")}
        report["source_pixels_sha256"] = hashlib.sha256(np.asarray(images["source"]).tobytes()).hexdigest()
        save()
        del visual
        gc.collect()
        pipe.transformer.to("cuda:0")
        torch.cuda.empty_cache()
        return images

    def run_case(pipe, name, prompt, seed, image):
        case = {"name": name, "seed": seed, "prompt": prompt}
        report["cases"].append(case)

        def on_step(_pipe, _step, _timestep, kwargs):
            if time.time() - START > BUDGET_SECONDS:
                raise TimeoutError("budget reached")
            return kwargs

        t = time.time()
        try:
            output = pipe(
                prompt=prompt, image=image, width=SIZE, height=SIZE, output_resolution=SIZE, use_kv_cache=True,
                generator=torch.Generator("cpu").manual_seed(seed), callback_on_step_end=on_step,
            ).images[0]
            output.save(f"{OUT}/{name}.png")
            case["file"] = f"{name}.png"
            case["outcome"] = "ok"
        except Exception as e:
            case["outcome"] = "error"
            case["error"] = repr(e)
        case["seconds"] = round(time.time() - t, 1)
        case["peak_gib"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
                            for i in range(torch.cuda.device_count())]
        print(name, case["outcome"], case["seconds"], case.get("error", ""), flush=True)
        save()
        gc.collect()
        torch.cuda.empty_cache()

    def main():
        t = time.time()
        pipe = build()
        report["setup_seconds"] = round(time.time() - t, 1)
        images = make_images(pipe)
        run_case(pipe, "source_raincoat_s1", EDITS["raincoat"], 1, images["source"])
        for seed in SEEDS:
            for edit, prompt in EDITS.items():
                for condition in ("vision", "both"):
                    if time.time() - START < BUDGET_SECONDS:
                        run_case(pipe, f"{condition}_{edit}_s{seed}", prompt, seed, images[condition])

    try:
        main()
    except Exception:
        report["fatal"] = traceback.format_exc()[-3000:]
        print(report["fatal"], flush=True)
    report["total_seconds"] = round(time.time() - START, 1)
    save()

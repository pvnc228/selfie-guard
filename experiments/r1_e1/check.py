# R1 E1 small check on Kaggle T4 x2: the same edits and seeds on a plain and a VAE-attacked copy of one
# generated face. Not a pilot: ten attempts on one photo cannot show a protection rate.
# The face and the attack are produced here, with the settings of e0.py and pgd_vae.py, so no image is
# uploaded. Results go to /kaggle/working, weights to /tmp.
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

BUDGET_SECONDS = 5100
SIZE = 768
EPS = 16 / 255
STEP = 0.02
ITERS = 200
OUT = "/kaggle/working"
START = time.time()

SUBJECT = ("Candid photograph of a fictional adult woman in her thirties wearing a grey sweater, "
           "standing on a city street, face clearly visible, natural daylight.")
EDITS = {
    "raincoat": "Change the person's upper clothing to a bright red raincoat. "
    "Keep the face, pose and background unchanged.",
    "snow": "Change the background to a snowy mountain village. Keep the person, face and clothing unchanged.",
}
SEEDS = [1, 2, 3, 4, 5]

report = {"model_rev": MODEL_REV, "gguf": [GGUF_REPO, GGUF_REV, GGUF_FILE], "diffusers": DIFFUSERS,
          "budget_seconds": BUDGET_SECONDS, "size": SIZE, "eps": EPS, "step": STEP, "iters": ITERS, "cases": []}


def save():
    with open(f"{OUT}/e1.json", "w") as f:
        json.dump(report, f, indent=1, default=str)


os.environ["HF_HOME"] = "/tmp/hf"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS}",
                f"transformers=={TRANSFORMERS}", "accelerate", "gguf"], check=True)

import torch  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor  # noqa: E402

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
    # Same steps as e0.py for the face and pgd_vae.py for the attack; see those files for the reasons.
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

    def to_tensor(image):
        return processor.preprocess(image.convert("RGBA"), width=SIZE, height=SIZE).unsqueeze(2).to("cuda:0", DTYPE)

    def latent(x):
        return (encode(x).latent_dist.mode() - mean) / std

    x = to_tensor(source)
    with torch.no_grad():
        clean = latent(x)
    torch.manual_seed(1)
    rgb = torch.tensor([1.0, 1.0, 1.0, 0.0], device="cuda:0", dtype=DTYPE).view(1, 4, 1, 1, 1)
    adv = (x + (torch.rand_like(x.float()).to(DTYPE) * 2 - 1) * EPS * rgb).clamp(-1, 1)
    t = time.time()
    for i in range(ITERS):
        step = STEP - (STEP - STEP / 100) / ITERS * i
        adv.requires_grad_(True)
        grad, = torch.autograd.grad(latent(adv).norm(), [adv])
        adv = adv.detach() - grad.sign() * step * rgb
        adv = torch.minimum(torch.maximum(adv, x - EPS), x + EPS).clamp(-1, 1)
    report["attack_seconds"] = round(time.time() - t, 1)
    pixels = ((adv[0, :3, 0].float().cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).round().astype(np.uint8)
    Image.fromarray(pixels, "RGB").save(f"{OUT}/protected.png")
    images = {k: Image.open(f"{OUT}/{k}.png").convert("RGB") for k in ("source", "protected")}
    difference = np.asarray(images["protected"], dtype=np.float64) - np.asarray(images["source"], dtype=np.float64)
    with torch.no_grad():
        final = latent(to_tensor(images["protected"]))
    report["inputs_sha256"] = {k: hashlib.sha256(open(f"{OUT}/{k}.png", "rb").read()).hexdigest() for k in images}
    report["latent_norm"] = [round(clean.norm().item(), 2), round(final.norm().item(), 2)]
    report["linf_8bit"] = int(np.abs(difference).max())
    report["psnr_db"] = round(float(10 * np.log10(255**2 / (difference**2).mean())), 2)
    print("attack", report["attack_seconds"], report["latent_norm"], report["psnr_db"], flush=True)
    save()
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
    case["peak_gib"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]
    print(name, case["outcome"], case["seconds"], case.get("error", ""), flush=True)
    save()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    t = time.time()
    pipe = build()
    report["setup_seconds"] = round(time.time() - t, 1)
    images = make_images(pipe)
    # Pairs in a fixed order, so that the budget never cuts one condition more than the other by a whole round.
    for seed in SEEDS:
        for edit, prompt in EDITS.items():
            for condition in ("source", "protected"):
                if time.time() - START < BUDGET_SECONDS:
                    run_case(pipe, f"{condition}_{edit}_s{seed}", prompt, seed, images[condition])


try:
    main()
except Exception:
    report["fatal"] = traceback.format_exc()[-3000:]
    print(report["fatal"], flush=True)
report["total_seconds"] = round(time.time() - START, 1)
save()

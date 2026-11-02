# R1 E2: PGD on the Qwen3-VL vision model alone and together with the VAE encoder, on a local GPU. Only the vision
# weights are read from the text encoder file. This script measures the feature change and its cost; whether edits
# break is measured separately. The losses and steps match experiments/r1_e2/check.py, which timed out on a T4.
import argparse
import hashlib
import importlib.util
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from diff_jpeg import jpeg
from diffusers import AutoencoderKLQwenImage21
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

spec = importlib.util.spec_from_file_location("e2_check", os.path.join(os.path.dirname(__file__), "check.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("out")
parser.add_argument("--modes", nargs="*", default=["vision", "both"])
parser.add_argument("--measure", nargs="*", default=[], help="existing files to compare with the image, no attack")
parser.add_argument("--iters", type=int, default=check.ITERS)
parser.add_argument("--views", choices=["all", "jpeg", "jpeg_only"], default="all",
                    help="with --eot: leave out the resize, or also the unchanged file")
parser.add_argument("--quality", nargs=2, type=int, default=[70, 95], help="with --eot: JPEG quality range, inclusive")
parser.add_argument("--scale", nargs=2, type=float, default=[0.5, 0.9], help="with --eot: resize scale range")
parser.add_argument("--target", choices=["none", "gray"], default="none", help="pull features to a target image")
parser.add_argument("--features", choices=["all", "pooled", "deep"], default="all", help="features the loss uses")
parser.add_argument("--eps", type=float, default=check.EPS, help="L-inf budget on the [-1, 1] scale")
parser.add_argument("--step", type=float, default=check.STEP, help="first step size; decays to step/100")
parser.add_argument("--eot", action="store_true", help="vision loss on a random JPEG or resize of the image each step")
parser.add_argument("--models", default="models/hf")
args = parser.parse_args()

torch.set_num_threads(min(6, torch.get_num_threads()))
DTYPE = torch.bfloat16
os.makedirs(args.out, exist_ok=True)
report = {**vars(args), "model_rev": check.MODEL_REV, "torch": torch.__version__, "dtype": str(DTYPE),
          "gpu": torch.cuda.get_device_name(0), "size": check.SIZE, "attacks": {}}


def save():
    with open(f"{args.out}/e2_attack.json", "w") as f:
        json.dump(report, f, indent=1)


base = snapshot_download(check.MODEL, revision=check.MODEL_REV, cache_dir=args.models,
                         allow_patterns=["vae/*", "text_encoder/config.json", "text_encoder/model.safetensors"])
with open(f"{base}/text_encoder/config.json") as f:
    config = json.load(f)["vision_config"]
visual = Qwen3VLVisionModel(Qwen3VLVisionConfig(**{k: v for k, v in config.items() if k not in ("dtype", "model_type")}))
with safe_open(f"{base}/text_encoder/model.safetensors", framework="pt") as f:
    prefix = next(k for k in f.keys() if ".visual." in k or k.startswith("visual.")).split("visual.")[0] + "visual."
    state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
visual.load_state_dict(state, strict=True)
del state
visual = visual.to("cuda:0", DTYPE).requires_grad_(False).eval()

vae = AutoencoderKLQwenImage21.from_pretrained(f"{base}/vae", torch_dtype=DTYPE).to("cuda:0")
vae.requires_grad_(False)
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to("cuda:0", DTYPE)
std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to("cuda:0", DTYPE)
processor = VaeImageProcessor(vae_scale_factor=vae.config.scale_factor_spatial)


def to_tensor(image):
    tensor = processor.preprocess(image.convert("RGBA"), width=check.SIZE, height=check.SIZE).unsqueeze(2)
    return tensor.to("cuda:0", DTYPE)


def latent(x):
    return (vae.encode(x).latent_dist.mode() - mean) / std


source = Image.open(args.image).convert("RGB")
report["source_pixels_sha256"] = hashlib.sha256(np.asarray(source).tobytes()).hexdigest()
x = to_tensor(source)
with torch.no_grad():
    clean_latent = latent(x)
    clean_vision = check.vision_features(visual, x[:, :3, 0])
rgb = torch.tensor([1.0, 1.0, 1.0, 0.0], device="cuda:0", dtype=DTYPE).view(1, 4, 1, 1, 1)


def shared_copy(image):
    # One of: the file as is, a JPEG of the --quality range, or a bicubic round trip through a --scale.
    kind = 1 if args.views == "jpeg_only" else torch.randint(2 if args.views == "jpeg" else 3, (1,)).item()
    if kind == 0:
        return image
    if kind == 1:
        return jpeg(image, torch.randint(args.quality[0], args.quality[1] + 1, (1,)).item()).to(image.dtype)
    size = int(image.shape[-1] * (args.scale[0] + (args.scale[1] - args.scale[0]) * torch.rand(1).item()))
    small = F.interpolate(image.float(), size=(size, size), mode="bicubic", antialias=True)
    return F.interpolate(small, size=image.shape[-2:], mode="bicubic", antialias=True).to(image.dtype)


with torch.no_grad():
    target_vision = check.vision_features(visual, torch.zeros_like(x[:, :3, 0]))  # mid gray on the [-1, 1] scale


def objective(view):
    # Larger is better. All variants are compared by check.vision_distance on the saved file.
    features = check.vision_features(visual, view)
    total = 0
    for i in {"all": [0, 1, 2, 3], "pooled": [0], "deep": [1, 2, 3]}[args.features]:
        reference = target_vision[i] if args.target == "gray" else clean_vision[i]
        distance = F.mse_loss(features[i], reference) / clean_vision[i].var()
        total = total + (-distance if args.target == "gray" else distance)
    return total


def attack(mode):
    torch.manual_seed(1)
    torch.cuda.reset_peak_memory_stats(0)
    adv = (x + (torch.rand_like(x.float()).to(DTYPE) * 2 - 1) * args.eps * rgb).clamp(-1, 1)
    result = {"losses": []}
    report["attacks"][mode] = result
    started = time.time()
    for i in range(args.iters):
        step = args.step - (args.step - args.step / 100) / args.iters * i
        adv.requires_grad_(True)
        view = adv[:, :3, 0]
        distance = objective(shared_copy(view) if args.eot else view)
        # The two losses have unrelated scales, so each gradient gets the same L1 weight before the sign.
        grad, = torch.autograd.grad(distance, [adv])
        direction = -grad / grad.abs().sum()
        entry = [i, round(distance.item(), 4)]
        if mode == "both":
            norm = latent(adv).norm()
            grad, = torch.autograd.grad(norm, [adv])
            direction = direction + grad / grad.abs().sum()
            entry.append(round(norm.item(), 2))
        if i % 10 == 0 or i == args.iters - 1:
            result["losses"].append(entry)
            result["seconds"] = round(time.time() - started, 1)
            save()
        adv = adv.detach() - direction.sign() * step * rgb
        adv = torch.minimum(torch.maximum(adv, x - args.eps), x + args.eps).clamp(-1, 1)
    result["seconds"] = round(time.time() - started, 1)
    result["seconds_per_iter"] = round(result["seconds"] / args.iters, 2)
    result["peak_gib"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 2)

    # Everything below is measured on the 8-bit file an attacker would actually receive.
    pixels = ((adv[0, :3, 0].float().cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).round().astype(np.uint8)
    path = f"{args.out}/protected_{mode}.png"
    Image.fromarray(pixels, "RGB").save(path)
    reread = Image.open(path).convert("RGB")
    difference = np.asarray(reread, dtype=np.float64) - np.asarray(source, dtype=np.float64)
    final = to_tensor(reread)
    with torch.no_grad():
        result["latent_norm"] = [round(clean_latent.norm().item(), 2), round(latent(final).norm().item(), 2)]
        result["vision_distance"] = round(check.vision_distance(visual, final[:, :3, 0], clean_vision).item(), 4)
    result["linf_8bit"] = int(np.abs(difference).max())
    result["psnr_db"] = round(float(10 * np.log10(255**2 / (difference**2).mean())), 2)
    result["sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()


for mode in args.modes:
    try:
        attack(mode)
    except torch.OutOfMemoryError as e:
        report["attacks"][mode]["error"] = repr(e)[:500]
        report["attacks"][mode]["peak_gib"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 2)
    torch.cuda.empty_cache()
    save()
report["measured"] = {}
for path in args.measure:
    final = to_tensor(Image.open(path).convert("RGB"))
    difference = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) - np.asarray(source, dtype=np.float64)
    with torch.no_grad():
        report["measured"][os.path.basename(path)] = {
            "vision_distance": round(check.vision_distance(visual, final[:, :3, 0], clean_vision).item(), 4),
            "latent_norm": round(latent(final).norm().item(), 2),
            "psnr_db": round(float(10 * np.log10(255**2 / (difference**2).mean())), 2),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        }
    save()
print(json.dumps(report, indent=1))

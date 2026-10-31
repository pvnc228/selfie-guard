# R1 E1: PGD on the Qwen-Image-2.1 VAE encoder only, after PhotoGuard's simple attack (MadryLab/photoguard
# 686bea75). Pushes the normalized latent towards zero. The Qwen3-VL path stays unattacked. This script
# measures the latent change and its cost only; whether edits break is measured separately.
import argparse
import json
import os
import time

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

MODEL = "Qwen/Qwen-Image-2.1-Turbo"
MODEL_REV = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("out")
parser.add_argument("--size", type=int, default=768)
parser.add_argument("--eps", type=float, default=16 / 255, help="L-inf budget on the [-1, 1] scale")
parser.add_argument("--step", type=float, default=0.02, help="first step size; decays to step/100")
parser.add_argument("--iters", type=int, default=3)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--models", default="models/hf")
args = parser.parse_args()

torch.set_num_threads(min(6, torch.get_num_threads()))
DTYPE = torch.bfloat16
os.makedirs(args.out, exist_ok=True)
report = {**vars(args), "model_rev": MODEL_REV, "torch": torch.__version__, "dtype": str(DTYPE),
          "gpu": torch.cuda.get_device_name(0)}

base = snapshot_download(MODEL, revision=MODEL_REV, allow_patterns=["vae/*"], cache_dir=args.models)
vae = AutoencoderKLQwenImage21.from_pretrained(f"{base}/vae", torch_dtype=DTYPE).to("cuda:0")
vae.requires_grad_(False)
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to("cuda:0", DTYPE)
std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to("cuda:0", DTYPE)
processor = VaeImageProcessor(vae_scale_factor=vae.config.scale_factor_spatial)


def to_tensor(image):
    # Same steps as QwenImage21Pipeline.__call__: RGBA, the shared preprocess, a frame axis.
    tensor = processor.preprocess(image.convert("RGBA"), width=args.size, height=args.size).unsqueeze(2)
    return tensor.to("cuda:0", DTYPE)


def latent(x):
    return (vae.encode(x).latent_dist.mode() - mean) / std


source = processor.resize(Image.open(args.image).convert("RGBA"), width=args.size, height=args.size)
x = to_tensor(source)
with torch.no_grad():
    clean = latent(x)

torch.manual_seed(args.seed)
torch.cuda.reset_peak_memory_stats(0)
rgb = torch.tensor([1.0, 1.0, 1.0, 0.0], device="cuda:0", dtype=DTYPE).view(1, 4, 1, 1, 1)  # alpha stays opaque
adv = (x + (torch.rand_like(x.float()).to(DTYPE) * 2 - 1) * args.eps * rgb).clamp(-1, 1)
losses = []
started = time.time()
for i in range(args.iters):
    step = args.step - (args.step - args.step / 100) / args.iters * i
    adv.requires_grad_(True)
    loss = latent(adv).norm()
    grad, = torch.autograd.grad(loss, [adv])
    losses.append(round(loss.item(), 2))
    adv = adv.detach() - grad.sign() * step * rgb
    adv = torch.minimum(torch.maximum(adv, x - args.eps), x + args.eps).clamp(-1, 1)
report["attack_seconds"] = round(time.time() - started, 1)
report["seconds_per_iter"] = round(report["attack_seconds"] / args.iters, 2)
report["attack_peak_gib"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 2)
report["losses"] = losses[:5] + losses[-5:]

# Everything below is measured on the 8-bit file an attacker would actually receive.
pixels = ((adv[0, :3, 0].float().cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).round().astype(np.uint8)
protected = Image.fromarray(pixels, "RGB")
protected.save(f"{args.out}/protected.png")
source.convert("RGB").save(f"{args.out}/source.png")
reread = Image.open(f"{args.out}/protected.png")
original = np.asarray(source.convert("RGB"), dtype=np.float64)
difference = np.asarray(reread, dtype=np.float64) - original
with torch.no_grad():
    final = latent(to_tensor(reread))
    report["clean_latent_norm"] = round(clean.norm().item(), 2)
    report["protected_latent_norm"] = round(final.norm().item(), 2)
    report["latent_rmse"] = round(((final - clean).float() ** 2).mean().sqrt().item(), 4)
report["linf_8bit"] = int(np.abs(difference).max())
report["psnr_db"] = round(float(10 * np.log10(255**2 / (difference**2).mean())), 2)

json.dump(report, open(f"{args.out}/e1.json", "w"), indent=1)
print(json.dumps(report, indent=1))

# CPU check of experiments/r1_e2/check.py without weights: the replacement preprocessor against the checkpoint's
# real one, and gradients through a small random Qwen3-VL vision model.
# Usage: python experiments/r1_e2/parity.py IMAGE [MODELS_DIR]
import importlib.util
import os
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import Qwen3VLProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

spec = importlib.util.spec_from_file_location("e2_check", os.path.join(os.path.dirname(__file__), "check.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

models = sys.argv[2] if len(sys.argv) > 2 else "models/hf"
base = snapshot_download(check.MODEL, revision=check.MODEL_REV, allow_patterns=["processor/*"], cache_dir=models)
real = Qwen3VLProcessor.from_pretrained(f"{base}/processor").image_processor
assert (real.patch_size, real.merge_size, real.temporal_patch_size) == (check.PATCH, check.MERGE, check.TEMPORAL)
assert (real.size["shortest_edge"], real.size["longest_edge"]) == (check.MIN_PIXELS, check.MAX_PIXELS)

source = Image.open(sys.argv[1]).convert("RGB")
for width, height in ((768, 768), (1024, 1024), (700, 500), (1000, 667)):
    image = source.resize((width, height), Image.BICUBIC)
    x = torch.tensor(np.asarray(image), dtype=torch.float32).permute(2, 0, 1)[None] / 255 * 2 - 1
    ours, grid = check.pixel_values(x)
    theirs = real(image, return_tensors="pt")
    assert grid.tolist() == theirs["image_grid_thw"].tolist()
    difference = (ours - theirs["pixel_values"]).abs()
    print(f"{width}x{height} grid {grid.tolist()} max {difference.max():.5f} mean {difference.mean():.5f}"
          f" (8-bit step {2 / 255:.5f})")

config = Qwen3VLVisionConfig(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=48,
                             patch_size=check.PATCH, spatial_merge_size=check.MERGE,
                             temporal_patch_size=check.TEMPORAL, deepstack_visual_indexes=[0, 1, 2],
                             num_position_embeddings=2304)
visual = Qwen3VLVisionModel(config).requires_grad_(False).eval()
x = torch.tensor(np.asarray(source.resize((256, 256))), dtype=torch.float32).permute(2, 0, 1)[None] / 255 * 2 - 1
with torch.no_grad():
    clean = check.vision_features(visual, x)
print("features", [tuple(f.shape) for f in clean])
assert check.vision_distance(visual, x, clean).item() == 0
adv = (x + 0.05 * torch.randn_like(x)).requires_grad_(True)
distance = check.vision_distance(visual, adv, clean)
grad, = torch.autograd.grad(distance, [adv])
print("distance", round(distance.item(), 4), "gradient nonzero share", round((grad != 0).float().mean().item(), 3))
assert distance.item() > 0 and (grad != 0).float().mean() > 0.9
print("ok")

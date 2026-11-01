# R1 E2: the file changes an image commonly meets after being shared, applied to a protected PNG.
# Usage: python experiments/r1_e2/transforms.py PROTECTED.png OUT_DIR
import os
import sys

from PIL import Image

image = Image.open(sys.argv[1]).convert("RGB")
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
image.save(f"{out}/jpeg90.jpg", quality=90)
image.save(f"{out}/jpeg75.jpg", quality=75)
half = image.resize((image.width // 2, image.height // 2), Image.BICUBIC)
half.resize(image.size, Image.BICUBIC).save(f"{out}/resize50.png")

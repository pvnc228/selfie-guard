# Differentiable approximation of baseline JPEG with 4:2:0 chroma, as Pillow writes by default: the quantization
# rounding is straight-through. Input and output are RGB tensors in [-1, 1], with sides divisible by 16.
import math

import torch
import torch.nn.functional as F

LUMA = [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]]
CHROMA = [[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99], [24, 26, 56, 99, 99, 99, 99, 99],
          [47, 66, 99, 99, 99, 99, 99, 99]] + [[99] * 8] * 4
RGB_TO_YCC = [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]


def _table(base, quality, like):
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return torch.floor((like.new_tensor(base) * scale + 50) / 100).clamp(1, 255)


def _dct_matrix(like):
    rows = [[(0.5 if u else math.sqrt(0.125)) * math.cos((2 * x + 1) * u * math.pi / 16) for x in range(8)]
            for u in range(8)]
    return like.new_tensor(rows)


def _quantize(plane, table):
    # plane: [B, 1, H, W], level-shifted to about [-128, 127].
    b, _, h, w = plane.shape
    blocks = plane.reshape(b, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)
    dct = _dct_matrix(plane)
    coefficients = dct @ blocks @ dct.T / table
    coefficients = coefficients + (coefficients.round() - coefficients).detach()
    blocks = dct.T @ (coefficients * table) @ dct
    return blocks.permute(0, 1, 3, 2, 4).reshape(b, 1, h, w)


def jpeg(x, quality):
    x = (x.float() + 1) * 127.5
    matrix = x.new_tensor(RGB_TO_YCC)
    ycc = torch.einsum("ij,bjhw->bihw", matrix, x)
    luma = _quantize(ycc[:, :1] - 128, _table(LUMA, quality, x)) + 128
    chroma = F.avg_pool2d(ycc[:, 1:], 2)
    table = _table(CHROMA, quality, x)
    chroma = torch.cat([_quantize(chroma[:, i:i + 1], table) for i in range(2)], dim=1)
    chroma = F.interpolate(chroma, scale_factor=2, mode="bilinear", align_corners=False)
    rgb = torch.einsum("ij,bjhw->bihw", torch.linalg.inv(matrix), torch.cat([luma, chroma], dim=1))
    # Clamping in the graph would zero the gradient of saturated pixels, so only the value is clamped.
    rgb = rgb + (rgb.clamp(0, 255) - rgb).detach()
    return rgb / 127.5 - 1

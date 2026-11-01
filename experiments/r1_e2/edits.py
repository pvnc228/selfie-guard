# R1 E2 robustness check on Kaggle T4 x2, edits only: the attack took about 25 s per iteration on a T4 (check.py), so
# the images come from experiments/r1_e2/pgd_both.py --eot and transforms.py on a local GPU through a private dataset.
# Where does the stop end: does the attack on the deepstack features stop edits on faces a and b after JPEG 85 and 80?
# Earlier versions of this script, with other inputs, are the E2 checks recorded in the journal.
# Not a pilot: two photos cannot show a protection rate. Results go to /kaggle/working, weights to /tmp.
import gc
import glob
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
SIZE = 768
INPUTS = {
    "a_source": "e88c9796877f75c602d73cf1849056954fe63ba04ea7d79d2a418456c8360fc3",
    "b_source": "eb9e1862b9726b3288c64793917c1c776211135d24bd205814c76d82524bc7d0",
    "a_deep_jpeg85": "e53badc73b89e00b76a751e3330413b901882bb4571ed6ec80df8ed7ccb46829",
    "a_deep_jpeg80": "e8016606d158a3cbdbeb781775257286e0ee019b2085f443934822fbf96cab1e",
    "b_deep_jpeg85": "30c64e97ec632683c73371f53221efd880c8ce40e77db963e5a25b39c84db85e",
    "b_deep_jpeg80": "6706fcfc938f56ec261537d6e60e064c2b2d219802f9914f68fbd62554bc9410",
}
CASES = [(f"{face}_deep_jpeg{quality}", seed) for face in "ab" for quality in (85, 80) for seed in (1, 2)]  # input, seed
OUT = "/kaggle/working"
START = time.time()

EDITS = {
    "raincoat": "Change the person's upper clothing to a bright red raincoat. "
    "Keep the face, pose and background unchanged.",
    "snow": "Change the background to a snowy mountain village. Keep the person, face and clothing unchanged.",
}

report = {"model_rev": MODEL_REV, "gguf": [GGUF_REPO, GGUF_REV, GGUF_FILE], "diffusers": DIFFUSERS,
          "budget_seconds": BUDGET_SECONDS, "size": SIZE, "inputs_sha256": INPUTS, "cases": []}


def save():
    with open(f"{OUT}/e2.json", "w") as f:
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


def load_inputs():
    images = {}
    for name, digest in INPUTS.items():
        path = glob.glob(f"/kaggle/input/**/{name}.*", recursive=True)[0]
        actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if actual != digest:
            raise ValueError(f"{name}: {actual}")
        images[name] = Image.open(path).convert("RGB")
    return images


def measure_embeds(pipe, images):
    # How much of the pixel attack reaches what the transformer actually reads.
    def embeds(image):
        with torch.no_grad():
            return pipe._get_qwen_prompt_embeds(EDITS["raincoat"], image=[image], device=torch.device("cuda:0"))[0].float()

    report["embeds"] = {}
    for name in INPUTS:
        if "_source" in name:
            continue
        clean_name = f"{name[0]}_source"
        clean = embeds(images[clean_name])
        report["embeds"][name] = {"clean_rms": round((clean**2).mean().sqrt().item(), 4),
                                  "rmse_vs_clean": round(((embeds(images[name]) - clean) ** 2).mean().sqrt().item(), 4)}
    save()


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
    images = load_inputs()
    t = time.time()
    pipe = build()
    report["setup_seconds"] = round(time.time() - t, 1)
    measure_embeds(pipe, images)
    for name, seed in CASES:
        for edit, prompt in EDITS.items():
            if time.time() - START < BUDGET_SECONDS:
                run_case(pipe, f"{name}_{edit}_s{seed}", prompt, seed, images[name])


try:
    main()
except Exception:
    report["fatal"] = traceback.format_exc()[-3000:]
    print(report["fatal"], flush=True)
report["total_seconds"] = round(time.time() - START, 1)
save()

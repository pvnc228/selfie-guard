# R1 E0: can Qwen-Image-2.1-Turbo make the edit at all without protection?
# Technical control on Kaggle T4 x2. Generated images and non-intimate edits only.
# No protection is tested here. Results go to /kaggle/working, weights to /tmp.
import gc
import json
import os
import resource
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

BUDGET_SECONDS = 3000
OUT = "/kaggle/working"
START = time.time()

SUBJECTS = {
    "a": "Candid photograph of a fictional adult woman in her thirties wearing a grey sweater, "
    "standing on a city street, face clearly visible, natural daylight.",
    "b": "Candid photograph of a fictional adult man in his forties wearing a navy jacket, "
    "standing in a park, face clearly visible, natural daylight.",
}
EDITS = {
    "raincoat": "Change the person's upper clothing to a bright red raincoat. "
    "Keep the face, pose and background unchanged.",
    "snow": "Change the background to a snowy mountain village. Keep the person, face and clothing unchanged.",
}
SEEDS = [1, 2, 3]

report = {"model_rev": MODEL_REV, "gguf": [GGUF_REPO, GGUF_REV, GGUF_FILE], "diffusers": DIFFUSERS,
          "budget_seconds": BUDGET_SECONDS, "cases": []}


def save():
    with open(f"{OUT}/e0.json", "w") as f:
        json.dump(report, f, indent=1, default=str)


os.environ["HF_HOME"] = "/tmp/hf"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
t = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS}",
                f"transformers=={TRANSFORMERS}", "accelerate", "gguf"], check=True)
report["install_seconds"] = round(time.time() - t, 1)

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
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor  # noqa: E402

report["torch"] = torch.__version__
report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
DTYPE = torch.bfloat16
report["dtype"] = str(DTYPE)


class TwoGpuPipeline(QwenImage21Pipeline):
    # Denoising on cuda:0; the text encoder is split over cuda:1 and CPU and returns to cuda:0.
    @property
    def _execution_device(self):
        return torch.device("cuda:0")


def timed(name, fn):
    t = time.time()
    result = fn()
    report[f"{name}_seconds"] = round(time.time() - t, 1)
    print(name, report[f"{name}_seconds"], flush=True)
    save()
    return result


def load():
    base = snapshot_download(MODEL, revision=MODEL_REV, ignore_patterns=["transformer/*.safetensors", "assets/*"])
    gguf = hf_hub_download(GGUF_REPO, GGUF_FILE, revision=GGUF_REV)
    with open(f"{base}/model_index.json") as f:
        sigmas = json.load(f)["sample_sigmas"]
    return base, gguf, sigmas


def dequantize_custom_params(model):
    # diffusers dequantizes GGUF weights only inside GGUFLinear. QwenImage21ZeroCenterRMSNorm reads its
    # weight directly and otherwise sees raw bytes (8192 instead of 4096 values).
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, GGUFLinear):
            continue
        for name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, GGUFParameter):
                setattr(module, name, torch.nn.Parameter(dequantize_gguf_tensor(param), requires_grad=False))
                names.append(f"{module_name}.{name}")
    return names


def free_transformer_during(vae_call, transformer):
    # VAE decode at 1024 px asks for 5 GiB more than a T4 has left beside the transformer.
    def wrapped(*args, **kwargs):
        transformer.to("cpu")
        torch.cuda.empty_cache()
        try:
            return vae_call(*args, **kwargs)
        finally:
            torch.cuda.empty_cache()
            transformer.to("cuda:0")
    return wrapped


def build(base, gguf, sigmas):
    transformer = QwenImage21Transformer2DModel.from_single_file(
        gguf, config=base, subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE), torch_dtype=DTYPE,
    )
    report["dequantized_params"] = dequantize_custom_params(transformer)
    transformer.to("cuda:0")
    vae = AutoencoderKLQwenImage21.from_pretrained(f"{base}/vae", torch_dtype=DTYPE).to("cuda:0")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        f"{base}/text_encoder", dtype=DTYPE, device_map="auto", max_memory={1: "13GiB", "cpu": "24GiB"},
    )
    report["text_encoder_devices"] = sorted({str(d) for d in text_encoder.hf_device_map.values()})
    vae.encode = free_transformer_during(vae.encode, transformer)
    vae.decode = free_transformer_during(vae.decode, transformer)
    return TwoGpuPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(f"{base}/scheduler"),
        vae=vae, text_encoder=text_encoder,
        processor=Qwen3VLProcessor.from_pretrained(f"{base}/processor"),
        transformer=transformer, sample_sigmas=sigmas,
    )


def run_case(pipe, name, prompt, seed, image=None, size=1024, kv_cache=True):
    case = {"name": name, "prompt": prompt, "seed": seed, "size": size, "kv_cache": kv_cache, "steps": []}
    report["cases"].append(case)
    last = [time.time()]

    def on_step(_pipe, step, _timestep, kwargs):
        now = time.time()
        case["steps"].append(round(now - last[0], 1))
        last[0] = now
        if now - START > BUDGET_SECONDS:
            raise TimeoutError("budget reached")
        return kwargs

    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    t = time.time()
    try:
        output = pipe(
            prompt=prompt, image=image, width=size, height=size, output_resolution=size, use_kv_cache=kv_cache,
            generator=torch.Generator("cpu").manual_seed(seed), callback_on_step_end=on_step,
        ).images[0]
        case["file"] = f"{name}.png"
        output.save(f"{OUT}/{name}.png")
        case["outcome"] = "ok"
    except Exception as e:
        output = None
        case["outcome"] = "error"
        case["error"] = repr(e)
        case["traceback"] = traceback.format_exc()[-1500:]
    case["seconds"] = round(time.time() - t, 1)
    case["peak_gib"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]
    case["peak_ram_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    print(name, case["outcome"], case["seconds"], case["peak_gib"], case.get("error", ""), flush=True)
    save()
    gc.collect()
    torch.cuda.empty_cache()
    return output


def main():
    base, gguf, sigmas = timed("download", load)
    pipe = timed("load", lambda: build(base, gguf, sigmas))
    sources = {}
    for key, prompt in SUBJECTS.items():
        source = run_case(pipe, f"source_{key}", prompt, seed=100)
        if source is not None:
            sources[key] = source
    if not sources:
        return
    # With Q8_0, 1024 px ran out of memory on a T4 with and without the KV cache. Take the first setting
    # that works here. 2048 px fails in the vision encoder (16 GiB allocation on cuda:1), so it is not retried.
    key, source = next(iter(sources.items()))
    edit, prompt = next(iter(EDITS.items()))
    settings = None
    for candidate in ({"size": 1024, "kv_cache": True}, {"size": 1024, "kv_cache": False},
                      {"size": 768, "kv_cache": True}):
        name = f"probe_{key}_{edit}_{candidate['size']}_{'cache' if candidate['kv_cache'] else 'nocache'}"
        if run_case(pipe, name, prompt, 1, image=source, **candidate) is not None:
            settings = candidate
            break
    report["edit_settings"] = settings
    if settings is None:
        return
    # One round per seed, so that the budget cuts seeds rather than a whole subject or edit.
    for seed in SEEDS:
        for key, source in sources.items():
            for edit, prompt in EDITS.items():
                if time.time() - START < BUDGET_SECONDS:
                    run_case(pipe, f"edit_{key}_{edit}_s{seed}", prompt, seed, image=source, **settings)


try:
    main()
except Exception:
    report["fatal"] = traceback.format_exc()[-3000:]
    print(report["fatal"], flush=True)
report["total_seconds"] = round(time.time() - START, 1)
save()

"""Local environment diagnostics without model imports or network requests."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess


def environment_report():
    packages = {}
    for name in ("torch", "diffusers", "transformers", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    gpu = {"status": "not_found"}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            gpu = {
                "status": "available" if result.returncode == 0 else "error",
                "returncode": result.returncode,
                "csv_columns": ["name", "memory.total [MiB]", "driver_version"],
                "csv": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except subprocess.TimeoutExpired:
            gpu = {"status": "timeout", "timeout_seconds": 10}
        except OSError as error:
            gpu = {"status": "error", "message": str(error)}

    return {
        "schema_version": 1,
        "protection_implemented": False,
        "python": platform.python_version(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "packages": packages,
        "nvidia_gpu": gpu,
    }


def main():
    parser = argparse.ArgumentParser(
        prog="selfie-guard",
        description="Research scaffold. Image protection is not implemented.",
    )
    parser.add_argument("command", choices=("doctor",), help="Inspect local environment")
    parser.parse_args()
    print(json.dumps(environment_report(), indent=2, ensure_ascii=True))
    return 0

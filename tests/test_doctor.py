"""Diagnostics must remain usable on CPU-only or unavailable-GPU hosts."""

import importlib.metadata
import platform
import subprocess
import unittest
from unittest.mock import patch

from selfie_guard.cli import environment_report


def setUpModule():
    # On Windows, Python 3.11 runs `ver` through subprocess on the first uname call
    # and caches the result; do that before the tests mock subprocess.run.
    platform.uname()


class DoctorTests(unittest.TestCase):
    def test_cpu_host_without_ml_packages_needs_no_gpu_process(self):
        with (
            patch("selfie_guard.cli.shutil.which", return_value=None),
            patch(
                "selfie_guard.cli.importlib.metadata.version",
                side_effect=importlib.metadata.PackageNotFoundError,
            ),
            patch("selfie_guard.cli.subprocess.run") as run,
        ):
            report = environment_report()
        run.assert_not_called()
        self.assertEqual(report["nvidia_gpu"]["status"], "not_found")
        self.assertTrue(all(value is None for value in report["packages"].values()))
        self.assertFalse(report["protection_implemented"])

    def test_gpu_failures_are_reported_without_crashing(self):
        failures = (
            subprocess.CompletedProcess([], 1, "GPU is lost", ""),
            subprocess.TimeoutExpired("nvidia-smi", 10),
            OSError("GPU tool unavailable"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                options = (
                    {"side_effect": failure}
                    if isinstance(failure, Exception)
                    else {"return_value": failure}
                )
                with (
                    patch("selfie_guard.cli.shutil.which", return_value="nvidia-smi"),
                    patch("selfie_guard.cli.subprocess.run", **options),
                ):
                    report = environment_report()
                self.assertIn(report["nvidia_gpu"]["status"], ("error", "timeout"))


if __name__ == "__main__":
    unittest.main()

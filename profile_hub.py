"""Optional: profile the submitted ONNX file on Qualcomm AI Hub.

The contest asks every team to use the same device if one is announced.
The public packet names a Qualcomm Arduino UNO Q, and says to use the closest
low-power device in the catalog if that board is missing.

Do this once, before the event:

    qai-hub configure --api_token YOUR_TOKEN_HERE

Then, after run_pipeline.py has written submission/student_model_int8.onnx:

    .\\edge-ai-env\\Scripts\\python.exe profile_hub.py --device "DEVICE NAME FROM THE CATALOG"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Profile the submitted model on Qualcomm AI Hub.")
    parser.add_argument("--model", default="submission/student_model_int8.onnx")
    parser.add_argument("--device", default="", help="Exact AI Hub device name. Leave empty to print the catalog.")
    parser.add_argument("--metrics", default="submission/metrics.json")
    args = parser.parse_args()

    try:
        import qai_hub as hub
    except ImportError:
        print("qai-hub is not installed. From the virtual environment run: pip install qai-hub")
        return

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"No model at {model_path}. Run run_pipeline.py first.")
        return

    if not args.device:
        print("No device name was given. Devices this account can see:")
        try:
            for device in hub.get_devices():
                print(f"  {device.name}")
        except Exception as error:
            print(f"Could not list devices. Check the API token with: qai-hub configure --api_token YOUR_TOKEN")
            print(error)
        print("Pick the announced contest device, or the closest low-power board, and rerun with --device.")
        return

    device = hub.Device(args.device)
    print(f"Compiling {model_path.name} for {args.device}. This waits in the free-tier queue.")
    compile_job = hub.submit_compile_job(
        model=str(model_path),
        device=device,
        input_specs={"input": (1, 3, 224, 224)},
    )
    compile_job.wait()
    profile_job = hub.submit_profile_job(model=compile_job.get_target_model(), device=device)
    profile_job.wait()
    profile = profile_job.download_profile()
    text = profile if isinstance(profile, str) else str(profile)
    Path("submission").mkdir(exist_ok=True)
    Path("submission/ai_hub_profile.txt").write_text(text, encoding="utf-8")
    print(text)

    metrics_path = Path(args.metrics)
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["ai_hub_profile_summary"] = f"Device {args.device}. Full text in submission/ai_hub_profile.txt."
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Updated {metrics_path}")


if __name__ == "__main__":
    main()

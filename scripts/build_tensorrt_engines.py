"""
Build TensorRT engines for StreamDiffusion pipelines (standalone script)

USAGE:
------
1. With a YAML/JSON config file (recommended):
    python build_tensorrt.py --config my_pipeline.yaml

2. With command-line arguments (overrides config file if both are used):
    python build_tensorrt.py \
        --model_id stabilityai/sd-turbo \
        --t_index_list 20,35,45 \
        --height 512 \
        --width 512 \
        --engine_dir /workspace/ComfyUI/engines/ \
        --controlnets '[{"model_id": "lllyasviel/control_v11p_sd15_canny"}]' \
        --lora_dict '{"lora1": 0.7}' \
        --ipadapters '[{"type": "regular"}]'

NOTES:
- Example YAML config files are available in the ../configs folder.
- CLI arguments override values in the config file if both are provided.
- For complex types (lists/dicts), pass JSON strings.
- The script prints a summary of the config before building.
- Engines are built in the directory you specify (default: engines).
"""
import argparse
import os
import sys
import json
import traceback

# StreamDiffusion can be installed via requirements.txt
from streamdiffusion import create_wrapper_from_config

def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engines for StreamDiffusion (robust config-driven CLI)")
    parser.add_argument("--config", type=str, default=None, help="YAML or JSON config file for pipeline (overrides CLI args if provided)")
    parser.add_argument("--model_id", type=str, help="Model ID or path to load (e.g. KBlueLeaf/kohaku-v2.1, stabilityai/sd-turbo)")
    parser.add_argument("--t_index_list", type=str, help="Comma-separated list of denoising steps (e.g. 20,35,45)")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--width", type=int, help="Image width")
    parser.add_argument("--engine_dir", type=str, help="Directory to save TensorRT engines (default: engines)")
    parser.add_argument("--controlnets", type=str, help="ControlNet config list as JSON string (see YAML example)")
    parser.add_argument("--lora_dict", type=str, help="LoRA dictionary as JSON, e.g. {\"lora1\": 0.7, \"lora2\": 1.0}")
    parser.add_argument("--ipadapters", type=str, help="IPAdapter config list as JSON string (see YAML example)")
    parser.add_argument(
        "--compile_engines_only",
        action="store_true",
        default=False,
        help="If set, only compile TensorRT engines without loading full models",
    )
    # Add more arguments as needed for other config keys
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    from streamdiffusion import load_config

    # Load config file if provided
    config = {}
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"ERROR: Failed to load config file: {e}")
            sys.exit(1)

    # Merge CLI args into config (CLI takes precedence)
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if key == "config" or value is None:
            continue
        # Handle special parsing for known complex types
        if key == "t_index_list":
            try:
                config[key] = [int(x.strip()) for x in value.split(",") if x.strip()]
            except Exception as e:
                print(f"ERROR: Invalid t_index_list: {value}")
                sys.exit(1)
        elif key in {"lora_dict", "controlnets", "ipadapters"}:
            try:
                config[key] = json.loads(value)
            except Exception as e:
                print(f"ERROR: Invalid JSON for {key}: {e}")
                sys.exit(1)
        elif key == "compile_engines_only":
            config[key] = bool(value)
        else:
            config[key] = value

    # Always set acceleration for this script
    config["acceleration"] = "tensorrt"

    # Print summary
    print("Building TensorRT engines with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    try:
        wrapper = create_wrapper_from_config(config)
        print(f"SUCCESS: TensorRT engine(s) built in {config.get('engine_dir', 'engines')}")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR: {e}\n{tb}")
        sys.exit(1)

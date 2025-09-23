# ComfyUI-StreamDiffusion Migration

## Code Strategy
To enable ControlNet and StreamDiffusion features in ComfyUI nodes, we are porting all necessary modules and wrappers from the research codebase. This includes:
- Migrating the `StreamDiffusionWrapper` and its dependencies (e.g., `StreamDiffusion`, `postprocess_image`, config loader) into local files in this repo.
- Porting ControlNet support by including the relevant pipeline and base classes, preprocessors, and orchestrators.
- Refactoring imports so all nodes use local modules, not installed packages.
- Each ComfyUI node (e.g., `cnet_trt_node.py`) will instantiate and use these wrappers for image2image and ControlNet inference.
- TensorRT acceleration, LoRA refit, and engine export logic will be integrated and tested in the node context.
- All configuration loading and workflow logic will be handled via local config utilities.

This approach ensures the ComfyUI nodes are self-contained, robust, and easy to maintain as the migration progresses.

## Project Goal
Port the native StreamDiffusion research pipeline and features to ComfyUI/Comfystream nodes.

## Key Features to Migrate
- SD1.5/SDTurbo
    - CNET + TensorRT
    - Batching (frame batching for performance)
    - IP Adapters + TensorRT
    - Stream V2V (not accelerated, deferred)
- SDXL (progress by Marco, deferred)
    - Not yet TRT accelerated

## Implementation Notes
- Each feature is implemented as a wrapper around UNET
- Parameter updates are consolidated in a "stream parameter updater" module
- Composable node design: flexible selection of wrappers per node
- Future optimization: single node with option to select wrapper, chainable in workflows
    - Example: Node 1 (CNET wrapper) → Node 2 (IP Adapter wrapper)

## Current Wrapper Options
- CNET + TensorRT
    - Node(s) follow ComfyUI pattern (separate ksampler node)
    - Engine building tracked separately
    - Add node to comfyui streampack
    - Update entrypoint.sh to build TRT engines
- IP Adapters + TensorRT

## Decisions
- Start with two separate nodes (CNET + TRT, IP Adapter + TRT)
- Use new ComfyUI custom node standard (see ComfyUI Discord @Prakarsh Kaushik)

## Task Breakdown
1. Research new ComfyUI custom node pattern
    - Check Discord, reach out to Ryan for details
2. Create new ComfyUI-StreamDiffusion repo
    - Add IP Adapter + TRT node, requisite ksampler, reuse existing ComfyUI SD nodes (vae, etc.)
    - Default to using TensorRT
3. Investigate how to compose wrappers in separate ComfyUI nodes
4. Add CNET + TensorRT node
    - Ensure batching and frame batching support
    - Add engine building scripts, update entrypoint.sh for TRT engine build

## Directory Structure
```
ComfyUI-StreamDiffusion/
│
├── nodes/                  # Custom ComfyUI nodes (main code)
│   ├── __init__.py
│   ├── ip_adapter_trt_node.py
│   ├── cnet_trt_node.py
│   └── ... (other wrappers)
│
├── scripts/                # Engine building, entrypoints, utilities
│   ├── build_trt_engine.py
│   ├── entrypoint.sh
│   └── ...
│
├── README.md               # Migration goals, setup, usage
├── requirements.txt        # Node dependencies (torch, tensorrt, comfyui, etc.)
└── setup.py                # (optional) For packaging
```

## Node Templates
- `nodes/ip_adapter_trt_node.py`: IP Adapter + TensorRT node
- `nodes/cnet_trt_node.py`: CNET + TensorRT node

## Engine Building
- `scripts/build_trt_engine.py`: Add export logic from `/workspace/StreamDiffusion` to build TensorRT engines
- `scripts/entrypoint.sh`: Automate engine building and node startup

## How to Use
1. Fill in the TODOs in node and script templates with StreamDiffusion logic
2. Register nodes in `nodes/__init__.py`
3. Build engines using `scripts/build_trt_engine.py`
4. Start ComfyUI and use the custom nodes

## Migration Steps
- Port core logic from `/workspace/StreamDiffusion` into node classes
- Move engine building logic into `scripts/`
- Expose parameters in the node interface
- Document progress and usage in this README

---

Share this repo and README with your team to coordinate migration and development.

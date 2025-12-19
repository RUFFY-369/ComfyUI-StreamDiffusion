
## Example Workflows

This folder contains example workflow files for ComfyUI-StreamDiffusion:

- **sd15_all_dynamic_params_wlora.json**: Demonstrates a workflow using dynamic parameters and LoRA models with Stable Diffusion v1.5. It loads images, applies LoRA conditioning, and previews results. Useful for experimenting with different LoRA weights and image inputs.

- **sd15_tensorrt_engine_build.json**: Shows a workflow for building and running a TensorRT-accelerated Stable Diffusion v1.5 pipeline. It includes configuration for model IDs, ControlNet, image preprocessing (canny), LoRA integration, and hardware acceleration settings.

These workflows are compatible with ComfyUI and can be used to test advanced features such as dynamic parameter adjustment, LoRA model blending, and TensorRT acceleration. To use, open the JSON files in ComfyUI or ComfyStream as appropriate.

---

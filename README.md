# StreamDiffusion Node for ComfyUI

A real-time StreamDiffusion pipeline node for ComfyUI with integrated ControlNet, IPAdapter, and LoRA support.

## Features
- Real-time, multi-control Stable Diffusion workflows
- Integrated ControlNet, IPAdapter, and LoRA support
- TensorRT acceleration for high performance
- Flexible configuration and batching
- Dynamically change parameter values at runtime via the streaming sampler node

## Installation

### Using ComfyUI Registry (Recommended)
```
comfy node install streamdiffusion
```

### Manual Installation
Clone this repository into your ComfyUI custom nodes directory:
```
git clone https://github.com/RUFFY-369/ComfyUI-StreamDiffusion.git
```
Install dependencies:
```
pip install -r requirements.txt
```

## Usage
After installation, launch ComfyUI. The StreamDiffusion node and its features will be available in the node menu.

## Requirements
- Python 3.10+
- NVIDIA GPU (for TensorRT acceleration)
- ComfyUI >= 1.0.0

## Support & Documentation
- [Documentation (README)](https://github.com/RUFFY-369/ComfyUI-StreamDiffusion#readme)
- [Bug Tracker](https://github.com/RUFFY-369/ComfyUI-StreamDiffusion/issues)

## License
This project is licensed under the terms of the LICENSE file in this repository.

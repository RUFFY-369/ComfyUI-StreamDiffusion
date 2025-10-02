import torch
# --- Modular ControlNet+TRT nodes ---
from streamdiffusion import create_wrapper_from_config
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

# 1. Config node
class ControlNetTRTConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Model and ControlNet fields
                "model_id": ("STRING", {"default": "Lykon/dreamshaper-8"}),
                "controlnet_model_id": ("STRING", {"default": "lllyasviel/control_v11p_sd15_canny"}),
                "conditioning_scale": ("FLOAT", {"default": 0.29, "min": 0.0, "max": 2.0, "step": 0.01}),
                "preprocessor": ("STRING", {"default": "canny"}),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "t_index_list": ("STRING", {"default": "20,35,45"}),
                "frame_buffer_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "warmup": ("INT", {"default": 10, "min": 0, "max": 100}),
                "acceleration": ("STRING", {"default": "tensorrt"}),
                "use_denoising_batch": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 2}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "use_lcm_lora": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "blurry, low quality, distorted, 3d render", "multiline": True, "tooltip": "Text prompt specifying undesired aspects to avoid in the generated image."}),
                "guidance_scale": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 20.0, "step": 0.01, "tooltip": "Controls the strength of the guidance. Higher values make the image more closely match the prompt."}),
                # IPAdapter fields
                "ipadapter_model_path": ("STRING", {"default": "/workspace/ComfyUI/models/ipadapter/ip-adapter-plus_sd15.bin"}),
                "image_encoder_path": ("STRING", {"default": "/workspace/ComfyUI/models/ipadapter/image_encoder"}),
                "style_image": ("IMAGE", {"tooltip": "Style image for IPAdapter conditioning."}),
                "ipadapter_scale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "ipadapter_enabled": ("BOOLEAN", {"default": True}),
                # ControlNet switch
                "use_controlnet": ("BOOLEAN", {"default": True, "tooltip": "Enable or disable ControlNet conditioning."}),
                    # Removed stray fragment; num_image_tokens is correctly defined and used in config
                "num_image_tokens": ("INT", {"default": 16, "min": 1, "max": 256, "tooltip": "Number of image tokens for conditioning."}),
                # Additional config fields for TRT
                "engine_dir": ("STRING", {"default": "/workspace/ComfyUI/engines/", "tooltip": "Directory for TensorRT engine files."}),
                "device": ("STRING", {"default": "cuda", "tooltip": "Device to run inference on (e.g., cuda, cpu)."}),
                "dtype": ("STRING", {"default": "float16", "tooltip": "Data type for inference (e.g., float16, float32)."}),
                "use_tiny_vae": ("BOOLEAN", {"default": True, "tooltip": "Use tiny VAE for faster inference."}),
                "cfg_type": ("STRING", {"default": "self", "tooltip": "Config type for advanced options."}),
                "delta": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Delta multiplier for virtual residual noise, affecting image diversity."}),
            }
        }
    RETURN_TYPES = ("MULTICONTROL_CONFIG",)
    FUNCTION = "build_config"
    CATEGORY = "ControlNet+IPAdapter"
    DESCRIPTION = "Builds a config dictionary for ControlNet and IPAdapter together."

    def build_config(self, **kwargs):
        # Validate t_index_list
        t_index_raw = kwargs.pop("t_index_list")
        try:
            t_index_list = [int(x.strip()) for x in t_index_raw.split(",") if x.strip()]
        except Exception as e:
            raise ValueError(f"Invalid t_index_list: {t_index_raw}. Must be a comma-separated list of integers.")
        if not t_index_list or any([not isinstance(x, int) or x < 0 for x in t_index_list]):
            raise ValueError(f"t_index_list must be a non-empty list of non-negative integers. Got: {t_index_list}")

        # Validate height/width
        height = kwargs.get("height", 512)
        if height is None or not isinstance(height, int) or height <= 0:
            raise ValueError(f"Invalid height: {height}. Must be a positive integer.")
        width = kwargs.get("width", 512)
        if width is None or not isinstance(width, int) or width <= 0:
            raise ValueError(f"Invalid width: {width}. Must be a positive integer.")

        # Build ControlNet config
        controlnet_config = []
        if kwargs.get("use_controlnet", True):
            controlnet_config.append({
                "model_id": kwargs["controlnet_model_id"],
                "preprocessor": kwargs["preprocessor"],
                "conditioning_scale": kwargs["conditioning_scale"],
                "enabled": True,
                "preprocessor_params": {
                    "low_threshold": kwargs["low_threshold"],
                    "high_threshold": kwargs["high_threshold"]
                },
            })
        # Build IPAdapter config
        ipadapter_config = []
        if kwargs.get("ipadapter_enabled", True):
            ipadapter_config.append({
                "ipadapter_model_path": kwargs["ipadapter_model_path"],
                "image_encoder_path": kwargs["image_encoder_path"],
                "style_image": kwargs["style_image"],
                "scale": kwargs["ipadapter_scale"],
                "enabled": True,
                "num_image_tokens": kwargs["num_image_tokens"],
            })

        # Build main config dict with all YAML params
        config = dict(
            model_id=kwargs["model_id"],
            controlnets=controlnet_config,
            ipadapters=ipadapter_config,
            height=height,
            width=width,
            t_index_list=t_index_list,
            frame_buffer_size=kwargs["frame_buffer_size"],
            warmup=kwargs["warmup"],
            acceleration=kwargs["acceleration"],
            use_denoising_batch=kwargs["use_denoising_batch"],
            device=kwargs["device"],
            dtype=kwargs["dtype"],
            engine_dir=kwargs["engine_dir"],
            use_tiny_vae=kwargs["use_tiny_vae"],
            cfg_type=kwargs["cfg_type"],
            delta=kwargs["delta"],
            seed=kwargs["seed"],
            num_inference_steps=kwargs["num_inference_steps"],
            use_lcm_lora=kwargs["use_lcm_lora"],
            prompt=kwargs["prompt"],
            negative_prompt=kwargs["negative_prompt"],
            guidance_scale=kwargs["guidance_scale"],
            output_type="pt",  # Always use tensor output for consistency
        )
        return (config,)

class ControlNetTRTModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("MULTICONTROL_CONFIG", {"tooltip": "Config dictionary for ControlNet+IPAdapter."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "ControlNet+TRT"
    DESCRIPTION = "Loads and initializes the ControlNet+TRT wrapper/model for img2img workflows."

    def load_model(self, config):
        # Use width and height from config, fallback to 512 if missing
        height = config.get("height", 512)
        width = config.get("width", 512)
        wrapper = create_wrapper_from_config(config)
        

        return ((wrapper, config, (height, width)),)


class ControlNetTRTUpdateParams:
    @classmethod
    def INPUT_TYPES(cls):
        template_json = 'e.g. ["prompt1", "prompt2"] or [["prompt1", 1.0], ["prompt2", 0.5]]'
        template_csv = 'e.g. prompt1, prompt2'
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Live model tuple (wrapper, config, resolution)."}),
            },
            "optional": {
                "prompt_list": ("STRING", {"tooltip": f"List of (prompt, weight) tuples as JSON or comma-separated. {template_json}; {template_csv}"}),
                "prompt_interpolation_method": ("STRING", {"default": "slerp", "tooltip": "Prompt interpolation method."}),
                "seed_list": ("STRING", {"tooltip": f"List of (seed, weight) tuples as JSON or comma-separated. e.g. [42, 43] or [[42, 1.0], [43, 0.5]]; 42, 43"}),
                "seed_interpolation_method": ("STRING", {"default": "linear", "tooltip": "Seed interpolation method."}),
                "guidance_scale": ("FLOAT", {"tooltip": "Guidance scale."}),
                "num_inference_steps": ("INT", {"tooltip": "Number of inference steps."}),
                "delta": ("FLOAT", {"tooltip": "Delta parameter."}),
                "t_index_list": ("STRING", {"tooltip": f"Timesteps list as JSON or comma-separated. e.g. [20,35,45]; 20,35,45"}),
                "ipadapter_enabled": ("BOOLEAN", {"default": None, "tooltip": "Enable or disable IPAdapter conditioning."}),
                "controlnet_enabled": ("BOOLEAN", {"default": None, "tooltip": "Enable or disable ControlNet conditioning."}),
                "image_preprocessing_config": ("STRING", {"tooltip": f"Image preprocessing config as JSON or comma-separated. e.g. [\"resize\", \"normalize\"]; resize, normalize"}),
                "image_postprocessing_config": ("STRING", {"tooltip": f"Image postprocessing config as JSON or comma-separated. e.g. [\"clip\"]; clip"}),
                "latent_preprocessing_config": ("STRING", {"tooltip": f"Latent preprocessing config as JSON or comma-separated. e.g. [\"scale\"]; scale"}),
                "latent_postprocessing_config": ("STRING", {"tooltip": f"Latent postprocessing config as JSON or comma-separated. e.g. [\"quantize\"]; quantize"}),
                "style_image_path": ("STRING", {"tooltip": "File path to new style image for IPAdapter."}),
                "ipadapter_scale": ("FLOAT", {"tooltip": "New IPAdapter scale."}),
                "preprocessor": ("STRING", {"tooltip": "New ControlNet preprocessor type."}),
                "preprocessor_params": ("DICT", {"tooltip": "New ControlNet preprocessor parameters (dict)."}),
                "normalize_prompt_weights": ("BOOLEAN", {"default": None, "tooltip": "Whether to normalize prompt weights dynamically."}),
                "normalize_seed_weights": ("BOOLEAN", {"default": None, "tooltip": "Whether to normalize seed weights dynamically."}),
                "negative_prompt": ("STRING", {"tooltip": "Negative prompt to use for blending (overrides config negative_prompt)."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "update_params"
    CATEGORY = "ControlNet+TRT"
    DESCRIPTION = "Dynamically update IPAdapter, ControlNet preprocessor, and prompt in the live model in one node."

    def update_params(self, model, prompt_list=None, prompt_interpolation_method=None, seed_list=None, seed_interpolation_method=None, guidance_scale=None, num_inference_steps=None, delta=None, t_index_list=None, ipadapter_enabled=None, controlnet_enabled=None, image_preprocessing_config=None, image_postprocessing_config=None, latent_preprocessing_config=None, latent_postprocessing_config=None, style_image_path=None, ipadapter_scale=None, preprocessor=None, preprocessor_params=None, normalize_prompt_weights=None, normalize_seed_weights=None, negative_prompt=None):
        import json
        wrapper, config, (height, width) = model
        update_kwargs = {}
        def parse_list(val):
            if val is None:
                return None
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return [x.strip() for x in val.split(",") if x.strip()]
            return val

        if prompt_list is not None:
            update_kwargs["prompt_list"] = parse_list(prompt_list)
        if prompt_interpolation_method is not None:
            update_kwargs["prompt_interpolation_method"] = prompt_interpolation_method
        if seed_list is not None:
            update_kwargs["seed_list"] = parse_list(seed_list)
        if seed_interpolation_method is not None:
            update_kwargs["seed_interpolation_method"] = seed_interpolation_method
        if guidance_scale is not None:
            update_kwargs["guidance_scale"] = guidance_scale
        if num_inference_steps is not None:
            update_kwargs["num_inference_steps"] = num_inference_steps
        if delta is not None:
            update_kwargs["delta"] = delta
        if t_index_list is not None:
            update_kwargs["t_index_list"] = [int(x) for x in parse_list(t_index_list)]
        if normalize_prompt_weights is not None:
            update_kwargs["normalize_prompt_weights"] = normalize_prompt_weights
        if normalize_seed_weights is not None:
            update_kwargs["normalize_seed_weights"] = normalize_seed_weights
        if negative_prompt is not None:
            update_kwargs["negative_prompt"] = negative_prompt
        if ipadapter_scale is not None or ipadapter_enabled is not None:
            ip_cfg = {}
            if ipadapter_scale is not None:
                ip_cfg["scale"] = ipadapter_scale
                if "ipadapters" in config and len(config["ipadapters"]):
                    config["ipadapters"][0]["scale"] = ipadapter_scale
            if ipadapter_enabled is not None:
                ip_cfg["enabled"] = ipadapter_enabled
                if "ipadapters" in config and len(config["ipadapters"]):
                    config["ipadapters"][0]["enabled"] = ipadapter_enabled
            update_kwargs["ipadapter_config"] = ip_cfg
        
        if image_preprocessing_config is not None:
            update_kwargs["image_preprocessing_config"] = parse_list(image_preprocessing_config)
        if image_postprocessing_config is not None:
            update_kwargs["image_postprocessing_config"] = parse_list(image_postprocessing_config)
        if latent_preprocessing_config is not None:
            update_kwargs["latent_preprocessing_config"] = parse_list(latent_preprocessing_config)
        if latent_postprocessing_config is not None:
            update_kwargs["latent_postprocessing_config"] = parse_list(latent_postprocessing_config)
        
        
        # Combine logic for updating controlnet_config so update_kwargs["controlnet_config"] is only set once
        controlnet_config_update_needed = False
        controlnet_config = config.get("controlnets", []).copy()
        if controlnet_config and isinstance(controlnet_config[0], dict):
            if controlnet_enabled is not None:
                controlnet_config[0]["enabled"] = bool(controlnet_enabled)
                controlnet_config_update_needed = True
            if preprocessor is not None:
                controlnet_config[0]["preprocessor"] = preprocessor
                controlnet_config_update_needed = True
            if preprocessor_params is not None:
                controlnet_config[0]["preprocessor_params"].update(preprocessor_params)
                controlnet_config_update_needed = True
            if controlnet_config_update_needed:
                update_kwargs["controlnet_config"] = controlnet_config
                config["controlnets"] = controlnet_config
        if update_kwargs and hasattr(wrapper, "update_stream_params"):
            wrapper.update_stream_params(**update_kwargs)
        return ((wrapper, config, (height, width)),)


class ControlNetTRTStreamingSampler:
    # Track which wrapper instances have been warmed up
    _warmed_wrappers = set()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "ControlNet+TRT wrapper/model tuple (wrapper, config, resolution)."}),
                "input_image": ("IMAGE", {"tooltip": "Input image for ControlNet conditioning."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt for generation."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Negative prompt for generation."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "ControlNet+TRT"
    DESCRIPTION = "Sampler for ControlNet+TRT. Runs warmup and inference for img2img."

    def generate(self, model, input_image, prompt, negative_prompt):
        wrapper, config, (height, width) = model
        import numpy as np
        from PIL import Image

            # IMPORTANT: Do NOT update config['prompt'] or config['negative_prompt'] here.
            # Only call wrapper.update_prompt / update_negative_prompt.
            # Updating config can cause unwanted resets and suppress IPAdapter effects.
            # This fix preserves IPAdapter style conditioning during dynamic prompt changes.
        # Dynamically update prompt and negative_prompt
        if (prompt is not None and prompt != config.get("prompt", "")) or (negative_prompt is not None and negative_prompt != config.get("negative_prompt", "")):
            print(f"[ControlNetTRTStreamingSampler] Updating prompt/negative_prompt via update_prompt (clear_blending=False)")
            # Use both prompt and negative_prompt if changed, else fallback to config value
            new_prompt = prompt if prompt is not None else config.get("prompt", "")
            new_negative_prompt = negative_prompt if negative_prompt is not None else config.get("negative_prompt", "")
            if hasattr(wrapper, "update_prompt"):
                wrapper.update_prompt(new_prompt, new_negative_prompt, clear_blending=False)

        # Convert input to PIL and resize
        if isinstance(input_image, torch.Tensor):
            img_np = input_image.squeeze().cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
            input_pil = Image.fromarray(img_np).convert("RGB").resize((width, height))
        elif isinstance(input_image, Image.Image):
            input_pil = input_image.convert("RGB").resize((width, height))
        else:
            raise ValueError("input_image must be a torch.Tensor or PIL.Image")

        warmup_count = config.get("warmup", 50)
        wrapper_id = id(wrapper)
        if wrapper_id not in self._warmed_wrappers:
            print(f"[ControlNetTRTStreamingSampler] Running warmup {warmup_count} times for new wrapper instance {wrapper_id}")
            for i in range(warmup_count):
                print(f"Running warmup inference {i+1}/{warmup_count}...")
                # Update control image for all configured ControlNets
                if hasattr(wrapper.stream, '_controlnet_module') and wrapper.stream._controlnet_module:
                    controlnet_count = len(wrapper.stream._controlnet_module.controlnets)
                    print(f"Updating control image for {controlnet_count} ControlNet(s) on incoming frame from stream")
                    for i in range(controlnet_count):
                        wrapper.update_control_image(i, input_pil)
                else:
                    print(f"process_video: No ControlNet module found for incoming frame")
                _ = wrapper(input_pil)
            self._warmed_wrappers.add(wrapper_id)
        else:
            print(f"[ControlNetTRTStreamingSampler] Warmup already done for wrapper instance {wrapper_id}, skipping.")

        # Inference
        # Update control image for all configured ControlNets
        if hasattr(wrapper.stream, '_controlnet_module') and wrapper.stream._controlnet_module:
            controlnet_count = len(wrapper.stream._controlnet_module.controlnets)
            print(f"Updating control image for {controlnet_count} ControlNet(s) on incoming frame from stream")
            for i in range(controlnet_count):
                wrapper.update_control_image(i, input_pil)
        else:
            print(f"process_video: No ControlNet module found for incoming frame")
        output_tensor = wrapper(input_pil)

        if isinstance(output_tensor, torch.Tensor):
            # If batched, take first image
            if output_tensor.dim() == 4:
                output_tensor = output_tensor[0]
            if output_tensor.dim() == 3:
                output_tensor = output_tensor.permute(1, 2, 0)
            # Add batch dimension for ComfyUI
            output_tensor = output_tensor.unsqueeze(0)
        else:
            # Fallback: convert PIL image to tensor
            output_tensor = to_tensor(output_tensor).permute(1,2,0).unsqueeze(0)
        return (output_tensor,)



import torch
import os
import folder_paths
from .streamdiffusionwrapper import StreamDiffusionWrapper
import inspect
import inspect
import os
import sys
import functools
from PIL import Image
from torchvision.transforms.functional import to_tensor

ENGINE_DIR = os.path.join(folder_paths.models_dir,
                          "tensorrt/StreamDiffusion-engines")
LIVE_PEER_CHECKPOINT_DIR = os.path.join(folder_paths.models_dir,
                                        "models/ComfyUI--models/checkpoints")

_wrapper_params = None
def _get_wrapper_params():
    """Return cached StreamDiffusionWrapper defaults with our overrides applied."""
    global _wrapper_params
    if _wrapper_params is None:
        print("[wrapper] collecting parameters …")
        params = inspect.signature(StreamDiffusionWrapper).parameters

        _wrapper_params = {name: p.default for name, p in params.items()}
        _wrapper_params["engine_dir"] = ENGINE_DIR
        _wrapper_params["output_type"] = "pt"

        # show what we ended up with
        print("[wrapper] defaults:", _wrapper_params)

    return _wrapper_params


def _dbg(msg, *, verbose):
    """Lightweight debug helper."""
    if verbose:
        print(msg, file=sys.stderr)

@functools.lru_cache(maxsize=1)
def get_engine_configs(*, verbose=False):
    """
    Scan the TensorRT engine directory and return a list of valid engine sets.

    A valid set = a folder that contains a sub-folder with `unet.engine`
                 **and** both `vae_encoder.engine` and `vae_decoder.engine`.
    """
    # Allow turning on verbosity via env var as well.
    verbose = verbose or bool(os.getenv("DEBUG_SD_ENGINES"))

    _dbg(f"[scan] ENGINE_DIR = {ENGINE_DIR}", verbose=verbose)

    if not os.path.exists(ENGINE_DIR):
        _dbg("[scan] directory does not exist – nothing to return", verbose=verbose)
        return []

    configs = []

    for parent_dir in sorted(os.listdir(ENGINE_DIR)):
        parent_path = os.path.join(ENGINE_DIR, parent_dir)
        _dbg(f"[scan] checking {parent_path}", verbose=verbose)

        if not os.path.isdir(parent_path):
            _dbg("        (skipped: not a directory)", verbose=verbose)
            continue

        has_unet = has_vae = False

        for subdir in sorted(os.listdir(parent_path)):
            subdir_path = os.path.join(parent_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            unet_path = os.path.join(subdir_path, "unet.engine")
            vae_enc = os.path.join(subdir_path, "vae_encoder.engine")
            vae_dec = os.path.join(subdir_path, "vae_decoder.engine")

            if os.path.exists(unet_path):
                has_unet = True
                _dbg(f"        found UNet  : {unet_path}", verbose=verbose)

            if os.path.exists(vae_enc) and os.path.exists(vae_dec):
                has_vae = True
                _dbg(f"        found VAE   : {vae_enc}, {vae_dec}", verbose=verbose)

        if has_unet and has_vae:
            _dbg(f"[scan] ✔ valid engine set: {parent_dir}", verbose=verbose)
            configs.append(parent_dir)
        else:
            _dbg(f"[scan] ✖ incomplete (UNet={has_unet}, VAE={has_vae})", verbose=verbose)

    _dbg(f"[scan] completed. {len(configs)} valid set(s) found.", verbose=verbose)
    return configs

def get_live_peer_checkpoints():
    """Get list of .safetensors files from LivePeer checkpoint directory"""

    if not os.path.exists(LIVE_PEER_CHECKPOINT_DIR):
        return []
    
    checkpoints = []
    for file in os.listdir(LIVE_PEER_CHECKPOINT_DIR):
        if file.endswith(".safetensors"):
            checkpoints.append(file)
            
    return checkpoints

#TODO: remove?
class StreamDiffusionTensorRTEngineLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "engine_name": (get_engine_configs(), ),
            }
        }

    RETURN_TYPES = ("SDMODEL",)
    FUNCTION = "load_engine"
    CATEGORY = "StreamDiffusion"

    def load_engine(self, engine_name):
        return (os.path.join(ENGINE_DIR, engine_name),)

#TODO: remove?
class StreamDiffusionLPCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_live_peer_checkpoints(), ),
            }
        }
    
    RETURN_TYPES = ("SDMODEL",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Loads a model from LivePeer checkpoint directory. Interchangeable with StreamDiffusionCheckpointLoader."

    def load_model(self, model_name):
        mod = os.path.join(LIVE_PEER_CHECKPOINT_DIR, model_name)
        return (mod,)

class StreamDiffusionLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA model to load."}),
                "strength": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "The strength scale for the LoRA model. Higher values result in greater influence of the LoRA on the output."}),
            },
            "optional": {
                "previous_loras": ("LORA_DICT", {"tooltip": "Optional dictionary of previously loaded LoRAs to which the new LoRA will be added. Use this to combine multiple LoRAs."}),
            }
        }

    RETURN_TYPES = ("LORA_DICT",)
    OUTPUT_TOOLTIPS = ("Dictionary of loaded LoRA models.",)
    FUNCTION = "load_lora"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Loads a LoRA (Low-Rank Adaptation) model and adds it to the existing LoRA dictionary for application to the pipeline."

    def load_lora(self, lora_name, strength, previous_loras=None):
        # Initialize with previous loras if provided
        lora_dict = {} if previous_loras is None else previous_loras.copy()
        
        # Add new lora to dictionary
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_dict[lora_path] = strength
        
        return (lora_dict,)

class StreamDiffusionVaeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), ),
            }
        }

    RETURN_TYPES = ("VAE_PATH",)
    FUNCTION = "load_vae"
    CATEGORY = "StreamDiffusion"

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        return (vae_path,)

class StreamDiffusionCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }

    RETURN_TYPES = ("SDMODEL",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Loads a model from the ComfyUI checkpoint directory. Interchangeable with StreamDiffusionLPCheckpointLoader."
    def load_checkpoint(self, checkpoint):
        checkpoint_path = folder_paths.get_full_path("checkpoints", checkpoint)
        return (checkpoint_path,)

class StreamDiffusionAdvancedConfig:
    @classmethod
    def INPUT_TYPES(s):
        defaults = _get_wrapper_params()
        
        return {
            "required": {
                # Acceleration settings
                "warmup": ("INT", {"default": defaults["warmup"], "min": 0, "max": 100, 
                    "tooltip": "The number of warmup steps to perform before actual inference. Increasing this may improve stability at the cost of speed."}),
                "do_add_noise": ("BOOLEAN", {"default": defaults["do_add_noise"], 
                    "tooltip": "Whether to add noise during denoising steps. Enable this to allow the model to generate diverse outputs."}),
                "use_denoising_batch": ("BOOLEAN", {"default": defaults["use_denoising_batch"], 
                    "tooltip": "Whether to use batch denoising for performance optimization."}),
                
                # Similarity filter settings
                "enable_similar_image_filter": ("BOOLEAN", {"default": defaults["enable_similar_image_filter"], 
                    "tooltip": "Enable filtering out images that are too similar to previous outputs."}),
                "similar_image_filter_threshold": ("FLOAT", {"default": defaults["similar_image_filter_threshold"], "min": 0.0, "max": 1.0, 
                    "tooltip": "Threshold determining how similar an image must be to previous outputs to be filtered out (0.0 to 1.0)."}),
                "similar_image_filter_max_skip_frame": ("INT", {"default": defaults["similar_image_filter_max_skip_frame"], "min": 0, "max": 100, 
                    "tooltip": "Maximum number of frames to skip when filtering similar images."}),
                
                # Device settings
                "device": (["cuda", "cpu"], {"default": defaults["device"], 
                    "tooltip": "Device to run inference on. CPU will be significantly slower."}),
                "dtype": (["float16", "float32"], {"default": "float16" if defaults["dtype"] == torch.float16 else "float32", 
                    "tooltip": "Data type for inference. float16 uses less memory but may be less precise."}),
                "device_ids": ("STRING", {"default": str(defaults["device_ids"] or ""), 
                    "tooltip": "Comma-separated list of device IDs for multi-GPU support. Leave empty for single GPU."}),
                "use_safety_checker": ("BOOLEAN", {"default": defaults["use_safety_checker"], 
                    "tooltip": "Enable safety checker to filter NSFW content. May impact performance."}),
                "engine_dir": ("STRING", {"default": defaults["engine_dir"], 
                    "tooltip": "Directory for TensorRT engine files when using tensorrt acceleration."}),
            }
        }

    RETURN_TYPES = ("ADVANCED_CONFIG",)
    OUTPUT_TOOLTIPS = ("Combined configuration settings for acceleration, similarity filtering, and device parameters.",)
    FUNCTION = "get_config"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures advanced settings for StreamDiffusion including acceleration, similarity filtering, and device settings."

    def get_config(self, warmup, do_add_noise, use_denoising_batch,
                  enable_similar_image_filter, similar_image_filter_threshold, 
                  similar_image_filter_max_skip_frame,
                  device, dtype, device_ids, use_safety_checker, engine_dir):
        
        # Convert device_ids string to list if provided
        device_ids = [int(x.strip()) for x in device_ids.split(",")] if device_ids.strip() else None
        
        # Combine all settings into a single config dictionary
        advanced_config = {
            # Acceleration settings
            "warmup": warmup,
            "do_add_noise": do_add_noise,
            "use_denoising_batch": use_denoising_batch,
            
            # Similarity filter settings
            "enable_similar_image_filter": enable_similar_image_filter,
            "similar_image_filter_threshold": similar_image_filter_threshold,
            "similar_image_filter_max_skip_frame": similar_image_filter_max_skip_frame,
            
            # Device settings
            "device": device,
            "dtype": torch.float32 if dtype == "float32" else torch.float16,
            "device_ids": device_ids,
            "use_safety_checker": use_safety_checker,
            "engine_dir": engine_dir,
        }
        
        return (advanced_config,)

class StreamDiffusionSampler:
    _cached_config = None  # Will store config
    _cached_params = None  # Will store params
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream_model": ("STREAM_MODEL", {"tooltip": "The configured StreamDiffusion model to use for generation."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "The text prompt to guide image generation."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt specifying undesired aspects to avoid in the generated image."}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "tooltip": "The number of denoising steps. More steps often yield better results but take longer."}),
                "guidance_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 20.0, "step": 0.01, "tooltip": "Controls the strength of the guidance. Higher values make the image more closely match the prompt."}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Delta multiplier for virtual residual noise, affecting image diversity."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The input image for image-to-image generation mode."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The generated image.",)
    FUNCTION = "generate"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Generates images using the configured StreamDiffusion model and specified prompts and settings."

    def generate(self, stream_model, prompt, negative_prompt, num_inference_steps, 
                guidance_scale, delta, image=None):
        
        #stream_model is a tuple of (model, config)
        model, config = stream_model
        
        current_params = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'delta': delta
        }

        needs_prepare = self._cached_params != current_params
        needs_warmup = self._cached_config != config
        
        if needs_prepare:
            self._cached_params = current_params
            model.prepare(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                delta=delta
            )
            
        if model.mode == "img2img" and image is not None:
            # Handle batch of images
            batch_size = image.shape[0]
            outputs = []
            
            for i in range(batch_size):
                # Convert from BHWC to BCHW format for model input
                image_tensor = image[i].permute(2, 0, 1).unsqueeze(0)
                # Warmup model
                if needs_warmup:
                    self._cached_config = config
                    for _ in range(model.batch_size - 1):
                        model(image=image_tensor)
                    needs_warmup = False
                    

                output = model(image=image_tensor)
                
                output = self.ensure_type_tensor(output)
    
                # Convert CHW → BHWC
                output_tensor = output.permute(1, 2, 0).unsqueeze(0)
                
                outputs.append(output_tensor)
            
            # Stack outputs
            output_tensor = torch.cat(outputs, dim=0)
            
        else:
            # Text to image generation
            output = model.txt2img()

            output = self.ensure_type_tensor(output)

            # Convert CHW → BHWC
            output_tensor = output.permute(1, 2, 0).unsqueeze(0)

        return (output_tensor,)

    def ensure_type_tensor(self, output):
        if isinstance(output, Image.Image):
                # PIL => Tensor [C, H, W]
            output = to_tensor(output)
        elif isinstance(output, torch.Tensor):
            pass  # already good
        else:
                # e.g. numpy array H × W × C
            output = torch.as_tensor(output).permute(2, 0, 1)
        return output

class StreamDiffusionConfigMixin:
    @staticmethod
    def get_optional_inputs():
        return {
            "opt_lora_dict": ("LORA_DICT", {"tooltip": "Optional dictionary of LoRA models to apply."}),
            "opt_advanced_config": ("ADVANCED_CONFIG", {"tooltip": "Optional advanced configuration for performance, filtering, and device settings."}),
        }

    @staticmethod
    def apply_optional_configs(config, opt_lora_dict=None, opt_advanced_config=None):
        if opt_lora_dict:
            config["lora_dict"] = opt_lora_dict

        if opt_advanced_config:
            config.update(opt_advanced_config)
        
        return config

class StreamDiffusionConfig(StreamDiffusionConfigMixin):
    @classmethod
    def INPUT_TYPES(s):
        defaults = _get_wrapper_params()
        return {
            "required": {
                "model": ("SDMODEL", {"tooltip": "The StreamDiffusion model to use for generation."}),
                "t_index_list": ("STRING", {"default": "39,35,30", "tooltip": "Comma-separated list of t_index values determining at which steps to output images."}),
                "mode": (["img2img", "txt2img"], {"default": defaults["mode"], "tooltip": "Generation mode: image-to-image or text-to-image. Note: txt2img requires cfg_type of 'none'"}),
                "width": ("INT", {"default": defaults["width"], "min": 64, "max": 2048, "tooltip": "The width of the generated images."}),
                "height": ("INT", {"default": defaults["height"], "min": 64, "max": 2048, "tooltip": "The height of the generated images."}),
                "acceleration": (["none", "xformers", "tensorrt"], {"default": "none", "tooltip": "Acceleration method to optimize performance."}),
                "frame_buffer_size": ("INT", {"default": defaults["frame_buffer_size"], "min": 1, "max": 16, "tooltip": "Size of the frame buffer for batch denoising. Increasing this can improve performance at the cost of higher memory usage."}),
                "use_tiny_vae": ("BOOLEAN", {"default": defaults["use_tiny_vae"], "tooltip": "Use a TinyVAE model for faster decoding with slight quality tradeoff."}),
                "cfg_type": (["none", "full", "self", "initialize"], {"default": defaults["cfg_type"], "tooltip": """Classifier-Free Guidance type to control how guidance is applied:
- none: No guidance, fastest but may reduce quality
- full: Full guidance on every step, highest quality but slowest
- self: Self-guidance using previous frame, good balance of speed/quality
- initialize: Only apply guidance on first frame, fast with decent quality"""}),
                "use_lcm_lora": ("BOOLEAN", {"default": True, "tooltip": "Enable use of LCM-LoRA for latent consistency."}),
                "seed": ("INT", {"default": defaults["seed"], "min": -1, "max": 100000000, "tooltip": "Seed for generation. Use -1 for random seed."}),
            },
            "optional": s.get_optional_inputs()
        }

    RETURN_TYPES = ("STREAM_MODEL",)
    OUTPUT_TOOLTIPS = ("The configured StreamDiffusion model.",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = """
This configures a model for StreamDiffusion. The model can be configured with or without acceleration. With TensorRT acceleration enabled, this node will run a TensorRT engine with the supplied parameters. 
If a suitable engine does not exist, it will be created. This can be used with any given checkpoint from either StreamDiffusionCheckpointLoader or StreamDiffusionLPCheckpointLoader or StreamDiffusionLPModelLoader.
    """

    def load_model(self, model, t_index_list, mode, width, height, acceleration, 
                  frame_buffer_size, use_tiny_vae, cfg_type, use_lcm_lora, seed,
                  **optional_configs):
        
        t_index_list = [int(x.strip()) for x in t_index_list.split(",")]
        defaults = _get_wrapper_params()
        
        # Build base configuration
        config = {
            "model_id_or_path": model,
            "t_index_list": t_index_list,
            "mode": mode,
            "width": width,
            "height": height,
            "acceleration": acceleration,
            "frame_buffer_size": frame_buffer_size,
            "use_tiny_vae": use_tiny_vae,
            "cfg_type": cfg_type,
            "use_lcm_lora": use_lcm_lora,
            "seed": seed,
            "engine_dir": defaults["engine_dir"],
            "output_type": defaults["output_type"]
        }

        # Apply optional configs using mixin method
        config = self.apply_optional_configs(config, **optional_configs)

        model_name = os.path.splitext(os.path.basename(model))[0] if os.path.isfile(model) else model.split('/')[-1]
        parent_name = f"{model_name}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--mode-{mode}--t_index-{len(t_index_list)}--buffer-{frame_buffer_size}"
        config["engine_dir"] = os.path.join(
            config["engine_dir"],
            parent_name
        )
        wrapper = (StreamDiffusionWrapper(**config), config)

        return (wrapper,)

#TODO: confirm this works well with container deployment
class StreamDiffusionPrebuiltConfig(StreamDiffusionConfigMixin):
    @classmethod
    def INPUT_TYPES(s):
        defaults = _get_wrapper_params()
        return {
            "required": {
                "engine_config": (get_engine_configs(), {"tooltip": "Select from available prebuilt engine configurations"}),
                "t_index_list": ("STRING", {"default": "39,35,30", "tooltip": "Comma-separated list of t_index values. Must match the number of steps used to build the engine."}),
                "mode": (["img2img", "txt2img"], {"default": defaults["mode"], "tooltip": "Generation mode: image-to-image or text-to-image"}),
                "frame_buffer_size": ("INT", {"default": defaults["frame_buffer_size"], "min": 1, "max": 16, "tooltip": "Size of the frame buffer. Must match what was used when building engines."}),
                "width": ("INT", {"default": defaults["width"], "min": 64, "max": 2048, "tooltip": "Must match the width used when building engines"}),
                "height": ("INT", {"default": defaults["height"], "min": 64, "max": 2048, "tooltip": "Must match the height used when building engines"}),
            },
            "optional": {
                "model": ("SDMODEL", ),
                **s.get_optional_inputs(),
            }
        }

    RETURN_TYPES = ("STREAM_MODEL",)
    FUNCTION = "configure_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures a model for StreamDiffusion using existing TensorRT engines."

    def configure_model(self, engine_config, t_index_list, mode, frame_buffer_size, width, height, model=None, **optional_configs):


        # Convert t_index_list from string to list of ints
        t_index_list = [int(x.strip()) for x in t_index_list.split(",")]
        
        # Get the engine directory path
        engine_dir = os.path.join(ENGINE_DIR, engine_config)
        
        defaults = _get_wrapper_params()
        
        # Build base configuration
        config = {
            "model_id_or_path": model if model is not None else engine_config,
            "mode": mode,
            "acceleration": "tensorrt",
            "frame_buffer_size": frame_buffer_size,
            "t_index_list": t_index_list,
            "width": width,
            "height": height,
            "use_denoising_batch": True,
            "use_tiny_vae": True,  # Assuming TinyVAE was used in engine building
            "output_type": defaults["output_type"]
        }

        # Apply optional configs using mixin method
        config = self.apply_optional_configs(config, **optional_configs)
        config["engine_dir"] = engine_dir
        wrapper = (StreamDiffusionWrapper(**config), config)
        return (wrapper,)

NODE_CLASS_MAPPINGS = {
    "StreamDiffusionConfig": StreamDiffusionConfig,
    "StreamDiffusionSampler": StreamDiffusionSampler,
    "StreamDiffusionPrebuiltConfig": StreamDiffusionPrebuiltConfig,
    "StreamDiffusionLoraLoader": StreamDiffusionLoraLoader,
    "StreamDiffusionAdvancedConfig": StreamDiffusionAdvancedConfig,
    "StreamDiffusionCheckpointLoader": StreamDiffusionCheckpointLoader,
    "StreamDiffusionTensorRTEngineLoader": StreamDiffusionTensorRTEngineLoader,
    "StreamDiffusionLPCheckpointLoader": StreamDiffusionLPCheckpointLoader,
    "ControlNetTRTConfig": ControlNetTRTConfig,
    "ControlNetTRTModelLoader": ControlNetTRTModelLoader,
    "ControlNetTRTStreamingSampler": ControlNetTRTStreamingSampler,
    "ControlNetTRTUpdateParams": ControlNetTRTUpdateParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusionConfig": "StreamDiffusionConfig",
    "StreamDiffusionSampler": "StreamDiffusionSampler",
    "StreamDiffusionPrebuiltConfig": "StreamDiffusionPrebuiltConfig",
    "StreamDiffusionLoraLoader": "StreamDiffusionLoraLoader",
    "StreamDiffusionAdvancedConfig": "StreamDiffusionAdvancedConfig",
    "StreamDiffusionCheckpointLoader": "StreamDiffusionCheckpointLoader",
    "StreamDiffusionTensorRTEngineLoader": "StreamDiffusionTensorRTEngineLoader",
    "StreamDiffusionLPCheckpointLoader": "StreamDiffusionLPCheckpointLoader",
    "ControlNetTRTConfig": "ControlNet + TRT Config",
    "ControlNetTRTModelLoader": "ControlNet + TRT Model Loader",
    "ControlNetTRTStreamingSampler": "ControlNet + TRT Streaming Sampler",
    "ControlNetTRTUpdateParams": "ControlNet + TRT Update Params",
}
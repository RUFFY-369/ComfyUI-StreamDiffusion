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
                "do_add_noise": ("BOOLEAN", {"default": True, "tooltip": "Whether to add noise during denoising steps. Enable this to allow the model to generate diverse outputs."}),
                "enable_similar_image_filter": ("BOOLEAN", {"default": False, "tooltip": "Enable filtering out images that are too similar to previous outputs."}),
                "similar_image_filter_threshold": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0, "tooltip": "Threshold for similar image filtering."}),
                "similar_image_filter_max_skip_frame": ("INT", {"default": 10, "min": 0, "max": 100, "tooltip": "Maximum number of frames to skip when filtering similar images."}),
                "use_safety_checker": ("BOOLEAN", {"default": False, "tooltip": "Enable safety checker to filter NSFW content. May impact performance."}),
                "seed": ("INT", {"default": 2}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "use_lcm_lora": ("BOOLEAN", {"default": True}),
                # LoRA support: JSON string, e.g. {"lora1": 0.7, "lora2": 1.0}
                "lora_dict_str": ("STRING", {"default": "", "multiline": True, "tooltip": "LoRA dictionary as JSON, e.g. {\"lora1\": 0.7, \"lora2\": 1.0}"}),
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
                "num_image_tokens": ("INT", {"default": 16, "min": 1, "max": 256, "tooltip": "Number of image tokens for conditioning."}),
                # Additional config fields for TRT
                "engine_dir": ("STRING", {"default": "/workspace/ComfyUI/engines/", "tooltip": "Directory for TensorRT engine files."}),
                "device": ("STRING", {"default": "cuda", "tooltip": "Device to run inference on (e.g., cuda, cpu)."}),
                "dtype": ("STRING", {"default": "float16", "tooltip": "Data type for inference (e.g., float16, float32)."}),
                "use_tiny_vae": ("BOOLEAN", {"default": True, "tooltip": "Use tiny VAE for faster inference."}),
                "cfg_type": ("STRING", {"default": "self", "tooltip": "Config type for advanced options."}),
                "delta": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Delta multiplier for virtual residual noise, affecting image diversity."}),
                # Engine build only flag
                "compile_engines_only": ("BOOLEAN", {"default": False, "tooltip": "If True, only compile TensorRT engines without loading the model. If False, compile and load for inference."}),
            }
        }
    RETURN_TYPES = ("MULTICONTROL_CONFIG",)
    FUNCTION = "build_config"
    CATEGORY = "ControlNet+IPAdapter"
    DESCRIPTION = "Builds a config dictionary for ControlNet and IPAdapter together."

    def build_config(self, **kwargs):
        import json
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
        ipadapter_enabled = kwargs.get("ipadapter_enabled", True)
        if ipadapter_enabled:
            ipadapter_config.append({
                "ipadapter_model_path": kwargs["ipadapter_model_path"],
                "image_encoder_path": kwargs["image_encoder_path"],
                "style_image": kwargs["style_image"],
                "scale": kwargs["ipadapter_scale"],
                "enabled": True,
                "num_image_tokens": kwargs["num_image_tokens"],
            })
        else:
            # If explicitly disabled, add a config with enabled=False for clarity
            ipadapter_config.append({
                "ipadapter_model_path": kwargs["ipadapter_model_path"],
                "image_encoder_path": kwargs["image_encoder_path"],
                "style_image": kwargs["style_image"],
                "scale": kwargs["ipadapter_scale"],
                "enabled": False,
                "num_image_tokens": kwargs["num_image_tokens"],
            })

        # Parse lora_dict_str (JSON) to dict, with error handling
        lora_dict_str = kwargs.get("lora_dict_str", "").strip()
        lora_dict = None
        if lora_dict_str:
            try:
                lora_dict = json.loads(lora_dict_str)
                if not isinstance(lora_dict, dict):
                    raise ValueError("lora_dict_str must be a JSON object (dictionary)")
            except Exception as e:
                raise ValueError(f"Invalid lora_dict_str: {e}\nExample: {{\"lora1\": 0.7, \"lora2\": 1.0}}")
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
            do_add_noise=kwargs["do_add_noise"],
            enable_similar_image_filter=kwargs["enable_similar_image_filter"],
            similar_image_filter_threshold=kwargs["similar_image_filter_threshold"],
            similar_image_filter_max_skip_frame=kwargs["similar_image_filter_max_skip_frame"],
            use_safety_checker=kwargs["use_safety_checker"],
            device=kwargs["device"],
            dtype=kwargs["dtype"],
            engine_dir=kwargs["engine_dir"],
            use_tiny_vae=kwargs["use_tiny_vae"],
            cfg_type=kwargs["cfg_type"],
            delta=kwargs["delta"],
            seed=kwargs["seed"],
            num_inference_steps=kwargs["num_inference_steps"],
            use_lcm_lora=kwargs["use_lcm_lora"],
            lora_dict=lora_dict,
            prompt=kwargs["prompt"],
            negative_prompt=kwargs["negative_prompt"],
            guidance_scale=kwargs["guidance_scale"],
            output_type="pt",  # Always use tensor output for consistency
            compile_engines_only=kwargs["compile_engines_only"],
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


class ControlNetTRTEngineCreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("MULTICONTROL_CONFIG", {"tooltip": "Config dictionary for ControlNet+IPAdapter."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "create_engine"
    CATEGORY = "ControlNet+TRT"
    DESCRIPTION = "Creates the ControlNet+TRT wrapper/model for img2img workflows, but returns nothing. Used only to pre-build TensorRT engines."
    OUTPUT_NODE = True

    def create_engine(self, config):
        # Use width and height from config, fallback to 512 if missing
        height = config.get("height", 512)
        width = config.get("width", 512)
        wrapper = create_wrapper_from_config(config)
        # No return value; just triggers engine build
        return ()


class ControlNetTRTStreamingSampler:
    # Track which wrapper instances have been warmed up
    _warmed_wrappers = set()

    @classmethod
    def INPUT_TYPES(cls):
        # Merge all dynamic params from ControlNetTRTUpdateParams
        template_json = 'e.g. ["prompt1", "prompt2"] or [["prompt1", 1.0], ["prompt2", 0.5]]'
        template_csv = 'e.g. prompt1, prompt2'
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "ControlNet+TRT wrapper/model tuple (wrapper, config, resolution)."}),
                "input_image": ("IMAGE", {"tooltip": "Input image for ControlNet conditioning."}),
            },
            "optional": {
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt for generation."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Negative prompt for generation."}),
                # prompt_list, prompt_interpolation_method, seed_list, and seed_interpolation_method are commented out because blending/interpolation of prompts and seeds is only supported in the realtime StreamDiffusion demo, where the backend is continuously generating frames and can update the blend live. In ComfyStream/ComfyUI, single-image workflows do not support these features out of the box. To enable blending/interpolation in ComfyStream, a custom integration is needed to update these parameters for each frame in a streaming or animation context.
                # "prompt_list": ("STRING", {"tooltip": "Change the prompt(s) used for generation. Accepts JSON or comma-separated list."}),
                # "prompt_interpolation_method": ("STRING", {"default": "slerp", "tooltip": "How to blend/interpolate multiple prompts (e.g. slerp, lerp)."}),
                # "seed_list": ("STRING", {"tooltip": "Change the seed(s) for generation. Accepts JSON or comma-separated list."}),
                # "seed_interpolation_method": ("STRING", {"default": "linear", "tooltip": "How to blend/interpolate multiple seeds (e.g. linear)."}),
                "guidance_scale": ("FLOAT", {"tooltip": "Controls how closely the image matches the prompt. Higher = more adherence."}),
                "num_inference_steps": ("INT", {"tooltip": "Number of denoising steps. More steps = better quality, slower."}),
                "delta": ("FLOAT", {"tooltip": "Delta parameter for diversity. Higher = more diverse outputs."}),
                "t_index_list": ("STRING", {"tooltip": "Timesteps for output. Accepts JSON or comma-separated list."}),
                # ControlNet major updatable fields
                "controlnet_model_id": ("STRING", {"tooltip": "Switch to a different ControlNet model."}),
                "conditioning_scale": ("FLOAT", {"tooltip": "ControlNet conditioning strength. Higher = stronger effect."}),
                "controlnet_enabled": ("BOOLEAN", {"default": None, "tooltip": "Enable or disable ControlNet conditioning."}),
                "preprocessor": ("STRING", {"tooltip": "Select the preprocessor type (e.g. canny, depth, hed, lineart, sharpen)."}),
                "conditioning_channels": ("INT", {"tooltip": "Number of conditioning channels for ControlNet (advanced)."}),
                "weight_type": ("STRING", {"tooltip": "ControlNet weight type (advanced, e.g. uniform, linear)."}),
                "image_path": ("STRING", {"tooltip": "Path to new control image for ControlNet."}),
                # IPAdapter major updatable fields
                "ipadapter_enabled": ("BOOLEAN", {"default": None, "tooltip": "Enable or disable IPAdapter conditioning."}),
                "ipadapter_scale": ("FLOAT", {"tooltip": "IPAdapter conditioning strength. Higher = stronger effect."}),
                "ipadapter_model_path": ("STRING", {"tooltip": "Switch to a different IPAdapter model."}),
                "image_encoder_path": ("STRING", {"tooltip": "Path to image encoder for IPAdapter."}),
                "num_image_tokens": ("INT", {"tooltip": "Number of image tokens for IPAdapter conditioning."}),
                "style_image_path": ("STRING", {"tooltip": "Path to new style image for IPAdapter."}),
                "ipadapter_weight_type": ("STRING", {"tooltip": "IPAdapter per-layer scaling method (e.g. uniform, linear)."}),
                # General
                "normalize_prompt_weights": ("BOOLEAN", {"default": None, "tooltip": "Normalize prompt weights for blending."}),
                "normalize_seed_weights": ("BOOLEAN", {"default": None, "tooltip": "Normalize seed weights for blending."}),
                # Pre/post-processing configs
                "image_preprocessing_config": ("STRING", {"tooltip": "Image preprocessing steps (e.g. resize, normalize). JSON or comma-separated."}),
                "image_postprocessing_config": ("STRING", {"tooltip": "Image postprocessing steps (e.g. clip). JSON or comma-separated."}),
                "latent_preprocessing_config": ("STRING", {"tooltip": "Latent preprocessing steps (e.g. scale). JSON or comma-separated."}),
                "latent_postprocessing_config": ("STRING", {"tooltip": "Latent postprocessing steps (e.g. quantize). JSON or comma-separated."}),
                # Preprocessor-specific params (all optional)
                "canny_low_threshold": ("INT", {"default": None, "tooltip": "Canny: Low threshold for edge detection."}),
                "canny_high_threshold": ("INT", {"default": None, "tooltip": "Canny: High threshold for edge detection."}),
                "depth_model_name": ("STRING", {"default": None, "tooltip": "Depth: Model name (e.g. MiDaS)."}),
                "depth_detect_resolution": ("INT", {"default": None, "tooltip": "Depth: Detection resolution."}),
                "depth_image_resolution": ("INT", {"default": None, "tooltip": "Depth: Output image resolution."}),
                "hed_safe": ("BOOLEAN", {"default": None, "tooltip": "HED: Enable safe mode for edge detection."}),
                "lineart_coarse": ("BOOLEAN", {"default": None, "tooltip": "Lineart: Enable coarse mode."}),
                "lineart_anime_style": ("BOOLEAN", {"default": None, "tooltip": "Lineart: Enable anime style mode."}),
                "sharpen_intensity": ("FLOAT", {"default": None, "tooltip": "Sharpen: Intensity of sharpening."}),
                "sharpen_unsharp_radius": ("FLOAT", {"default": None, "tooltip": "Sharpen: Unsharp mask radius."}),
                "sharpen_edge_enhancement": ("FLOAT", {"default": None, "tooltip": "Sharpen: Edge enhancement factor."}),
                "sharpen_detail_boost": ("FLOAT", {"default": None, "tooltip": "Sharpen: Detail boost factor."}),
                "sharpen_noise_reduction": ("FLOAT", {"default": None, "tooltip": "Sharpen: Noise reduction factor."}),
                "sharpen_multi_scale": ("BOOLEAN", {"default": None, "tooltip": "Sharpen: Enable multi-scale sharpening."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "ControlNet+TRT"
    DESCRIPTION = "Sampler for ControlNet+TRT. Runs warmup and inference for img2img."

    def generate(self, model, input_image, **kwargs):
        wrapper, config, (height, width) = model
        import numpy as np
        from PIL import Image

        # --- DYNAMIC PARAM UPDATE LOGIC (from ControlNetTRTUpdateParams) ---
        update_kwargs = {}
        def parse_list(val):
            if val is None:
                return None
            import json
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return [x.strip() for x in val.split(",") if x.strip()]
            return val

        # Collect all dynamic params: use sampler override if set, else config value
        # prompt_list is commented out for now. See above for explanation.
        # prompt_list, prompt_interpolation_method, seed_list, and seed_interpolation_method are commented out for now. See above for explanation.
        for key in [
            # "prompt_list",  # See comment above
            # "prompt_interpolation_method",  # See comment above
            # "seed_list",  # See comment above
            # "seed_interpolation_method",  # See comment above
            "guidance_scale", "num_inference_steps", "delta", "t_index_list",
            "normalize_prompt_weights", "normalize_seed_weights", "negative_prompt",
            "image_preprocessing_config", "image_postprocessing_config", "latent_preprocessing_config", "latent_postprocessing_config"
        ]:
            val = kwargs.get(key, None)
            if val is None:
                val = config.get(key, None)
            # Special robust handling for t_index_list
            if key == "t_index_list":
                parsed = None
                if val is not None:
                    if isinstance(val, str):
                        # Try to parse as JSON or CSV
                        try:
                            parsed = [int(x) for x in parse_list(val)]
                        except Exception:
                            parsed = None
                    elif isinstance(val, list):
                        parsed = [int(x) for x in val if isinstance(x, (int, float, str)) and str(x).strip()]
                    else:
                        parsed = None
                if not parsed or not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed) or len(parsed) == 0:
                    raise ValueError("t_index_list must be a non-empty list of integers (e.g. '20,35,45') from config or sampler node. Got: {}".format(val))
                update_kwargs[key] = parsed
            elif key == "num_inference_steps":
                # Validate num_inference_steps is a positive int
                nsteps = None
                if val is not None:
                    try:
                        nsteps = int(val)
                    except Exception:
                        nsteps = None
                if nsteps is None or nsteps <= 0:
                    raise ValueError("num_inference_steps must be a positive integer from config or sampler node. Got: {}".format(val))
                update_kwargs[key] = nsteps
            elif val is not None:
                if key.endswith("_list") or key.endswith("_config"):
                    update_kwargs[key] = parse_list(val)
                else:
                    update_kwargs[key] = val

        # IPAdapter config
        ip_cfg = {}
        ipadapter_enabled = None
        for key in [
            "ipadapter_scale", "ipadapter_enabled", "ipadapter_model_path", "image_encoder_path", "num_image_tokens", "style_image_path", "ipadapter_weight_type"
        ]:
            val = kwargs.get(key, None)
            if val is None and "ipadapters" in config and len(config["ipadapters"]):
                val = config["ipadapters"][0].get(key if key != "ipadapter_scale" else "scale", None)
            if key == "ipadapter_enabled":
                ipadapter_enabled = val
            if val is not None:
                ip_cfg[key if key != "ipadapter_scale" else "scale"] = val
                if "ipadapters" in config and len(config["ipadapters"]):
                    config["ipadapters"][0][key if key != "ipadapter_scale" else "scale"] = val
        # Actually enable/disable IPAdapter in config
        if ipadapter_enabled is not None:
            if "ipadapters" in config and len(config["ipadapters"]):
                config["ipadapters"][0]["enabled"] = bool(ipadapter_enabled)
            ip_cfg["enabled"] = bool(ipadapter_enabled)
        if ip_cfg:
            update_kwargs["ipadapter_config"] = ip_cfg

        # ControlNet config
        controlnet_config_update_needed = False
        controlnet_config = config.get("controlnets", []).copy()
        preprocessor = kwargs.get("preprocessor", None)
        if preprocessor is None and controlnet_config and "preprocessor" in controlnet_config[0]:
            preprocessor = controlnet_config[0]["preprocessor"]
        preproc_type = preprocessor
        preproc_param_map = {
            "canny": ["canny_low_threshold", "canny_high_threshold"],
            "depth": ["depth_model_name", "depth_detect_resolution", "depth_image_resolution"],
            "hed": ["hed_safe"],
            "lineart": ["lineart_coarse", "lineart_anime_style"],
            "sharpen": ["sharpen_intensity", "sharpen_unsharp_radius", "sharpen_edge_enhancement", "sharpen_detail_boost", "sharpen_noise_reduction", "sharpen_multi_scale"],
        }
        preproc_params = {}
        if preproc_type in preproc_param_map:
            for param in preproc_param_map[preproc_type]:
                val = kwargs.get(param, None)
                if val is None and controlnet_config and "preprocessor_params" in controlnet_config[0]:
                    val = controlnet_config[0]["preprocessor_params"].get(param.split('_', 1)[1] if '_' in param else param, None)
                if val is not None:
                    key = param.split('_', 1)[1] if '_' in param else param
                    preproc_params[key] = val
        if controlnet_config and isinstance(controlnet_config[0], dict):
            for key in ["controlnet_model_id", "conditioning_scale", "controlnet_enabled", "preprocessor", "conditioning_channels", "weight_type", "image_path"]:
                val = kwargs.get(key, None)
                if val is None:
                    if key == "controlnet_model_id":
                        val = controlnet_config[0].get("model_id", None)
                    else:
                        val = controlnet_config[0].get(key, None)
                if val is not None:
                    if key == "controlnet_model_id":
                        controlnet_config[0]["model_id"] = val
                    elif key == "conditioning_scale":
                        controlnet_config[0]["conditioning_scale"] = val
                    elif key == "controlnet_enabled":
                        controlnet_config[0]["enabled"] = bool(val)
                    elif key == "preprocessor":
                        controlnet_config[0]["preprocessor"] = val
                    else:
                        controlnet_config[0][key] = val
                    controlnet_config_update_needed = True
            if preproc_params:
                controlnet_config[0]["preprocessor_params"].update(preproc_params)
                controlnet_config_update_needed = True
            if controlnet_config_update_needed:
                update_kwargs["controlnet_config"] = controlnet_config
                config["controlnets"] = controlnet_config

        if update_kwargs and hasattr(wrapper, "update_stream_params"):
            wrapper.update_stream_params(**update_kwargs)

        # --- END DYNAMIC PARAM UPDATE LOGIC ---

        # Prompt/neg prompt for backward compatibility
        prompt = kwargs.get("prompt", None)
        negative_prompt = kwargs.get("negative_prompt", None)
        if (prompt is not None and prompt != config.get("prompt", "")) or (negative_prompt is not None and negative_prompt != config.get("negative_prompt", "")):
            print(f"[ControlNetTRTStreamingSampler] Updating prompt/negative_prompt via update_prompt (clear_blending=False)")
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

        if hasattr(wrapper.stream, '_controlnet_module') and wrapper.stream._controlnet_module:
            controlnet_count = len(wrapper.stream._controlnet_module.controlnets)
            print(f"Updating control image for {controlnet_count} ControlNet(s) on incoming frame from stream")
            for i in range(controlnet_count):
                wrapper.update_control_image(i, input_pil)
        else:
            print(f"process_video: No ControlNet module found for incoming frame")
        output_tensor = wrapper(input_pil)

        if isinstance(output_tensor, torch.Tensor):
            if output_tensor.dim() == 4:
                output_tensor = output_tensor[0]
            if output_tensor.dim() == 3:
                output_tensor = output_tensor.permute(1, 2, 0)
            output_tensor = output_tensor.unsqueeze(0)
        else:
            output_tensor = to_tensor(output_tensor).permute(1,2,0).unsqueeze(0)
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "ControlNetTRTConfig": ControlNetTRTConfig,
    "ControlNetTRTModelLoader": ControlNetTRTModelLoader,
    "ControlNetTRTStreamingSampler": ControlNetTRTStreamingSampler,
    "ControlNetTRTEngineCreator": ControlNetTRTEngineCreator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNetTRTConfig": "ControlNet + TRT Config",
    "ControlNetTRTModelLoader": "ControlNet + TRT Model Loader",
    "ControlNetTRTStreamingSampler": "ControlNet + TRT Streaming Sampler",
    "ControlNetTRTEngineCreator": "ControlNet + TRT Engine Creator",
}

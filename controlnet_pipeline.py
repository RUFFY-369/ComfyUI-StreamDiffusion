

# import os
# import yaml
# from PIL import Image
# from typing import Optional, List, Dict, Any

# # Local utility: load_config
# def load_config(config_file: str) -> dict:
#     with open(config_file, 'r') as f:
#         return yaml.safe_load(f)

# # Local utility: create_wrapper_from_config
# def create_wrapper_from_config(config: dict):
#     from streamdiffusion import StreamDiffusionWrapper
#     return StreamDiffusionWrapper(
#         model_id_or_path=config['model_id'],
#         t_index_list=config.get('t_index_list', [1]),
#         width=config.get('width', 512),
#         height=config.get('height', 512),
#         use_controlnet=True,
#         controlnet_config=config.get('controlnets', None),
#         acceleration=config.get('acceleration', 'tensorrt'),
#         frame_buffer_size=config.get('frame_buffer_size', 1),
#         do_add_noise=config.get('do_add_noise', True),
#         cfg_type=config.get('cfg_type', 'self'),
#         # Add more config params as needed
#     )

# class ControlNetPipeline:
#     """
#     Multi-ControlNet StreamDiffusion pipeline for ComfyUI node integration.
#     """
#     def __init__(self, config_file: str):
#         self.config_file = config_file
#         self.wrapper = None
#         self.warmup_steps = 10
#         self._warmed_up = False
#         self._setup_pipeline()

#     def _setup_pipeline(self):
#         config_data = load_config(self.config_file)
#         self.wrapper = create_wrapper_from_config(config_data)
#         self.warmup_steps = config_data.get('warmup', 10)

#     def process_image(self, image: Image.Image) -> Image.Image:
#         """
#         Process a single image through the multi-ControlNet pipeline.
#         """
#         if not self._warmed_up and self.warmup_steps > 0:
#             for i in range(self.warmup_steps):
#                 if hasattr(self.wrapper, 'update_control_image_efficient'):
#                     self.wrapper.update_control_image_efficient(image)
#                 _ = self.wrapper(image)
#             self._warmed_up = True
#         if hasattr(self.wrapper, 'update_control_image_efficient'):
#             self.wrapper.update_control_image_efficient(image)
#         output = self.wrapper(image)
#         if isinstance(output, Image.Image):
#             return output
#         elif isinstance(output, (list, tuple)) and isinstance(output[0], Image.Image):
#             return output[0]
#         else:
#             # Try to convert numpy/tensor to PIL
#             try:
#                 from torchvision.transforms.functional import to_pil_image
#                 return to_pil_image(output)
#             except Exception:
#                 return output

#     def update_controlnet_strength(self, index: int, strength: float):
#         """
#         Dynamically update ControlNet strength.
#         """
#         if hasattr(self.wrapper, 'update_controlnet_scale'):
#             self.wrapper.update_controlnet_scale(index, strength)

#     def update_stream_params(self, guidance_scale: float = None, delta: float = None, num_inference_steps: int = None):
#         """
#         Dynamically update StreamDiffusion parameters during inference.
#         """
#         if hasattr(self.wrapper, 'stream') and hasattr(self.wrapper.stream, 'update_stream_params'):
#             self.wrapper.stream.update_stream_params(
#                 guidance_scale=guidance_scale,
#                 delta=delta,
#                 num_inference_steps=num_inference_steps
#             )

# """
# Script to build TensorRT engines from StreamDiffusion models.
# This script is currently commented out/not required for ControlNet TRT node testing.
# """

# # def build_trt_engine(model_path, output_path):
# #     """
# #     Build and export TensorRT engine from a StreamDiffusion model.
# #     TODO: Port export logic from StreamDiffusion repo.
# #     """
# #     # Example: Use torch2trt or custom export logic
# #     # from torch2trt import torch2trt
# #     # model = torch.load(model_path)
# #     # engine = torch2trt(model, ...)
# #     # with open(output_path, 'wb') as f:
# #     #     f.write(engine.serialize())
# #     pass
# #
# # if __name__ == "__main__":
# #     import argparse
# #     parser = argparse.ArgumentParser(description="Build TensorRT engine from StreamDiffusion model")
# #     parser.add_argument("--model", type=str, required=True, help="Path to model file")
# #     parser.add_argument("--output", type=str, required=True, help="Path to output engine file")
# #     args = parser.parse_args()
# #     build_trt_engine(args.model, args.output)

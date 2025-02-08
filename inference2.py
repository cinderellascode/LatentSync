import cv2
import numpy as np
from gfpgan import GFPGANer
from codeformer import CodeFormer

def apply_superres(frame, method):
    """Apply super-resolution to the frame using the specified method."""
    if method == "GFPGAN":
        restorer = GFPGANer(model_path="gfpgan_model.pth", upscale=2)
        _, _, output = restorer.enhance(frame, has_aligned=False, only_center_face=False)
    elif method == "CodeFormer":
        restorer = CodeFormer(model_path="codeformer_model.pth", upscale=2)
        output = restorer.enhance(frame)
    else:
        raise ValueError(f"Unsupported super-resolution method: {method}")
    return output

def main():
    # Existing code to load input video and process frames
    input_video = cv2.VideoCapture(args.input_video)
    output_video = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    while True:
        ret, frame = input_video.read()
        if not ret:
            break

        # Generate the lipsynced subframe (existing logic)
        subframe = generate_subframe(frame)

        # Check resolution ratio
        input_res = frame.shape[:2]
        output_res = subframe.shape[:2]
        ratio = input_res[0] / output_res[0]  # Assuming height is the primary dimension

        # Apply super-resolution if necessary
        if ratio > 1.0 and args.superres:
            subframe = apply_superres(subframe, args.superres)

        # Combine subframe with the original frame (existing logic)
        output_frame = combine_subframe(frame, subframe)

        # Write the output frame
        output_video.write(output_frame)

    input_video.release()
    output_video.release()
    
    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_video", type=str, required=True)
parser.add_argument("--output_video", type=str, required=True)
parser.add_argument("--superres", type=str, choices=["GFPGAN", "CodeFormer"], default=None)
args = parser.parse_args()


# import cv2
# import numpy as np
# from gfpgan import GFPGANer
# from codeformer import CodeFormer

# def apply_superres(frame, method):
#     """Apply super-resolution to the frame using the specified method."""
#     if method == "GFPGAN":
#         restorer = GFPGANer(model_path="gfpgan_model.pth", upscale=2)
#         _, _, output = restorer.enhance(frame, has_aligned=False, only_center_face=False)
#     elif method == "CodeFormer":
#         restorer = CodeFormer(model_path="codeformer_model.pth", upscale=2)
#         output = restorer.enhance(frame)
#     else:
#         raise ValueError(f"Unsupported super-resolution method: {method}")
#     return output

# def main():
#     # Existing code to load input video and process frames
#     input_video = cv2.VideoCapture(args.input_video)
#     output_video = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

#     while True:
#         ret, frame = input_video.read()
#         if not ret:
#             break

#         # Generate the lipsynced subframe (existing logic)
#         subframe = generate_subframe(frame)

#         # Check resolution ratio
#         input_res = frame.shape[:2]
#         output_res = subframe.shape[:2]
#         ratio = input_res[0] / output_res[0]  # Assuming height is the primary dimension

#         # Apply super-resolution if necessary
#         if ratio > 1.0 and args.superres:
#             subframe = apply_superres(subframe, args.superres)

#         # Combine subframe with the original frame (existing logic)
#         output_frame = combine_subframe(frame, subframe)

#         # Write the output frame
#         output_video.write(output_frame)

#     input_video.release()
#     output_video.release()
    
    
    
# import argparse
# from omegaconf import OmegaConf
# import torch
# from diffusers import AutoencoderKL, DDIMScheduler
# from latentsync.models.unet import UNet3DConditionModel
# from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
# from diffusers.utils.import_utils import is_xformers_available
# from accelerate.utils import set_seed
# from latentsync.whisper.audio2feature import Audio2Feature
# from PIL import Image
# import numpy as np
# import cv2

# # Super-resolution functions
# def apply_gfpgan(input_frame, output_subframe):
#     from gfpgan import GFPGANer
#     restorer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
#     _, _, restored_frame = restorer.enhance(output_subframe, has_aligned=False, only_center_face=False, paste_back=True)
#     return restored_frame

# def apply_codeformer(input_frame, output_subframe):
#     from codeformer import CodeFormer
#     restorer = CodeFormer()
#     restored_frame = restorer.enhance(output_subframe)
#     return restored_frame

# def main(config, args):
#     # Check if the GPU supports float16
#     is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
#     dtype = torch.float16 if is_fp16_supported else torch.float32

#     print(f"Input video path: {args.video_path}")
#     print(f"Input audio path: {args.audio_path}")
#     print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

#     scheduler = DDIMScheduler.from_pretrained("configs")

#     if config.model.cross_attention_dim == 768:
#         whisper_model_path = "checkpoints/whisper/small.pt"
#     elif config.model.cross_attention_dim == 384:
#         whisper_model_path = "checkpoints/whisper/tiny.pt"
#     else:
#         raise NotImplementedError("cross_attention_dim must be 768 or 384")

#     audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

#     vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
#     vae.config.scaling_factor = 0.18215
#     vae.config.shift_factor = 0

#     unet, _ = UNet3DConditionModel.from_pretrained(
#         OmegaConf.to_container(config.model),
#         args.inference_ckpt_path,  # load checkpoint
#         device="cpu",
#     )

#     unet = unet.to(dtype=dtype)

#     # set xformers
#     if is_xformers_available():
#         unet.enable_xformers_memory_efficient_attention()

#     pipeline = LipsyncPipeline(
#         vae=vae,
#         audio_encoder=audio_encoder,
#         unet=unet,
#         scheduler=scheduler,
#     ).to("cuda")

#     if args.seed != -1:
#         set_seed(args.seed)
#     else:
#         torch.seed()

#     print(f"Initial seed: {torch.initial_seed()}")

#     # Run the pipeline
#     pipeline(
#         video_path=args.video_path,
#         audio_path=args.audio_path,
#         video_out_path=args.video_out_path,
#         video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
#         num_frames=config.data.num_frames,
#         num_inference_steps=args.inference_steps,
#         guidance_scale=args.guidance_scale,
#         weight_dtype=dtype,
#         width=config.data.resolution,
#         height=config.data.resolution,
#     )

#     # Apply super-resolution if needed
#     if args.superres:
#         input_frame = cv2.imread(args.video_path)  # Load input frame
#         output_subframe = cv2.imread(args.video_out_path)  # Load output subframe

#         # Calculate resolution ratio
#         input_height, input_width, _ = input_frame.shape
#         output_height, output_width, _ = output_subframe.shape
#         resolution_ratio = (input_height * input_width) / (output_height * output_width)

#         if resolution_ratio > 1:  # Output resolution is poorer
#             print(f"Applying super-resolution using {args.superres}...")
#             if args.superres == "GFPGAN":
#                 restored_frame = apply_gfpgan(input_frame, output_subframe)
#             elif args.superres == "CodeFormer":
#                 restored_frame = apply_codeformer(input_frame, output_subframe)
#             else:
#                 raise ValueError("Invalid superres option. Choose 'GFPGAN' or 'CodeFormer'.")

#             # Save the restored frame
#             cv2.imwrite(args.video_out_path, restored_frame)
#             print(f"Super-resolution applied and saved to {args.video_out_path}.")
#         else:
#             print("Output resolution is sufficient. Super-resolution not applied.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
#     parser.add_argument("--inference_ckpt_path", type=str, required=True)
#     parser.add_argument("--video_path", type=str, required=True)
#     parser.add_argument("--audio_path", type=str, required=True)
#     parser.add_argument("--video_out_path", type=str, required=True)
#     parser.add_argument("--inference_steps", type=int, default=20)
#     parser.add_argument("--guidance_scale", type=float, default=1.0)
#     parser.add_argument("--seed", type=int, default=1247)
#     parser.add_argument("--superres", type=str, choices=["GFPGAN", "CodeFormer"], help="Apply super-resolution using GFPGAN or CodeFormer")
#     args = parser.parse_args()

#     config = OmegaConf.load(args.unet_config_path)

#     main(config, args)


# # # Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# # #
# # # Licensed under the Apache License, Version 2.0 (the "License");
# # # you may not use this file except in compliance with the License.
# # # You may obtain a copy of the License at
# # #
# # #     http://www.apache.org/licenses/LICENSE-2.0
# # #
# # # Unless required by applicable law or agreed to in writing, software
# # # distributed under the License is distributed on an "AS IS" BASIS,
# # # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # # See the License for the specific language governing permissions and
# # # limitations under the License.

# # import argparse
# # from omegaconf import OmegaConf
# # import torch
# # from diffusers import AutoencoderKL, DDIMScheduler
# # from latentsync.models.unet import UNet3DConditionModel
# # from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
# # from diffusers.utils.import_utils import is_xformers_available
# # from accelerate.utils import set_seed
# # from latentsync.whisper.audio2feature import Audio2Feature
# # from PIL import Image
# # import numpy as np
# # import cv2

# # # Super-resolution functions
# # def apply_gfpgan(input_frame, output_subframe):
# #     from gfpgan import GFPGANer
# #     restorer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
# #     _, _, restored_frame = restorer.enhance(output_subframe, has_aligned=False, only_center_face=False, paste_back=True)
# #     return restored_frame

# # def apply_codeformer(input_frame, output_subframe):
# #     from codeformer import CodeFormer
# #     restorer = CodeFormer()
# #     restored_frame = restorer.enhance(output_subframe)
# #     return restored_frame

# # def main(config, args):
# #     # Check if the GPU supports float16
# #     is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
# #     dtype = torch.float16 if is_fp16_supported else torch.float32

# #     print(f"Input video path: {args.video_path}")
# #     print(f"Input audio path: {args.audio_path}")
# #     print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

# #     scheduler = DDIMScheduler.from_pretrained("configs")

# #     if config.model.cross_attention_dim == 768:
# #         whisper_model_path = "checkpoints/whisper/small.pt"
# #     elif config.model.cross_attention_dim == 384:
# #         whisper_model_path = "checkpoints/whisper/tiny.pt"
# #     else:
# #         raise NotImplementedError("cross_attention_dim must be 768 or 384")

# #     audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

# #     vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
# #     vae.config.scaling_factor = 0.18215
# #     vae.config.shift_factor = 0

# #     unet, _ = UNet3DConditionModel.from_pretrained(
# #         OmegaConf.to_container(config.model),
# #         args.inference_ckpt_path,  # load checkpoint
# #         device="cpu",
# #     )

# #     unet = unet.to(dtype=dtype)

# #     # set xformers
# #     if is_xformers_available():
# #         unet.enable_xformers_memory_efficient_attention()

# #     pipeline = LipsyncPipeline(
# #         vae=vae,
# #         audio_encoder=audio_encoder,
# #         unet=unet,
# #         scheduler=scheduler,
# #     ).to("cuda")

# #     if args.seed != -1:
# #         set_seed(args.seed)
# #     else:
# #         torch.seed()

# #     print(f"Initial seed: {torch.initial_seed()}")

# #     # Run the pipeline
# #     pipeline(
# #         video_path=args.video_path,
# #         audio_path=args.audio_path,
# #         video_out_path=args.video_out_path,
# #         video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
# #         num_frames=config.data.num_frames,
# #         num_inference_steps=args.inference_steps,
# #         guidance_scale=args.guidance_scale,
# #         weight_dtype=dtype,
# #         width=config.data.resolution,
# #         height=config.data.resolution,
# #     )

# #     # Apply super-resolution if needed
# #     if args.superres:
# #         input_frame = cv2.imread(args.video_path)  # Load input frame
# #         output_subframe = cv2.imread(args.video_out_path)  # Load output subframe

# #         # Calculate resolution ratio
# #         input_height, input_width, _ = input_frame.shape
# #         output_height, output_width, _ = output_subframe.shape
# #         resolution_ratio = (input_height * input_width) / (output_height * output_width)

# #         if resolution_ratio > 1:  # Output resolution is poorer
# #             print(f"Applying super-resolution using {args.superres}...")
# #             if args.superres == "GFPGAN":
# #                 restored_frame = apply_gfpgan(input_frame, output_subframe)
# #             elif args.superres == "CodeFormer":
# #                 restored_frame = apply_codeformer(input_frame, output_subframe)
# #             else:
# #                 raise ValueError("Invalid superres option. Choose 'GFPGAN' or 'CodeFormer'.")

# #             # Save the restored frame
# #             cv2.imwrite(args.video_out_path, restored_frame)
# #             print(f"Super-resolution applied and saved to {args.video_out_path}.")
# #         else:
# #             print("Output resolution is sufficient. Super-resolution not applied.")


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
# #     parser.add_argument("--inference_ckpt_path", type=str, required=True)
# #     parser.add_argument("--video_path", type=str, required=True)
# #     parser.add_argument("--audio_path", type=str, required=True)
# #     parser.add_argument("--video_out_path", type=str, required=True)
# #     parser.add_argument("--inference_steps", type=int, default=20)
# #     parser.add_argument("--guidance_scale", type=float, default=1.0)
# #     parser.add_argument("--seed", type=int, default=1247)
# #     parser.add_argument("--superres", type=str, choices=["GFPGAN", "CodeFormer"], help="Apply super-resolution using GFPGAN or CodeFormer")
# #     args = parser.parse_args()

# #     config = OmegaConf.load(args.unet_config_path)

# #     main(config, args)
    
    
# # # Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# # #
# # # Licensed under the Apache License, Version 2.0 (the "License");
# # # you may not use this file except in compliance with the License.
# # # You may obtain a copy of the License at
# # #
# # #     http://www.apache.org/licenses/LICENSE-2.0
# # #
# # # Unless required by applicable law or agreed to in writing, software
# # # distributed under the License is distributed on an "AS IS" BASIS,
# # # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # # See the License for the specific language governing permissions and
# # # limitations under the License.

# # import argparse
# # from omegaconf import OmegaConf
# # import torch
# # from diffusers import AutoencoderKL, DDIMScheduler
# # from latentsync.models.unet import UNet3DConditionModel
# # from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
# # from diffusers.utils.import_utils import is_xformers_available
# # from accelerate.utils import set_seed
# # from latentsync.whisper.audio2feature import Audio2Feature


# # def main(config, args):
# #     # Check if the GPU supports float16
# #     is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
# #     dtype = torch.float16 if is_fp16_supported else torch.float32

# #     print(f"Input video path: {args.video_path}")
# #     print(f"Input audio path: {args.audio_path}")
# #     print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

# #     scheduler = DDIMScheduler.from_pretrained("configs")

# #     if config.model.cross_attention_dim == 768:
# #         whisper_model_path = "checkpoints/whisper/small.pt"
# #     elif config.model.cross_attention_dim == 384:
# #         whisper_model_path = "checkpoints/whisper/tiny.pt"
# #     else:
# #         raise NotImplementedError("cross_attention_dim must be 768 or 384")

# #     audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

# #     vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
# #     vae.config.scaling_factor = 0.18215
# #     vae.config.shift_factor = 0

# #     unet, _ = UNet3DConditionModel.from_pretrained(
# #         OmegaConf.to_container(config.model),
# #         args.inference_ckpt_path,  # load checkpoint
# #         device="cpu",
# #     )

# #     unet = unet.to(dtype=dtype)

# #     # set xformers
# #     if is_xformers_available():
# #         unet.enable_xformers_memory_efficient_attention()

# #     pipeline = LipsyncPipeline(
# #         vae=vae,
# #         audio_encoder=audio_encoder,
# #         unet=unet,
# #         scheduler=scheduler,
# #     ).to("cuda")

# #     if args.seed != -1:
# #         set_seed(args.seed)
# #     else:
# #         torch.seed()

# #     print(f"Initial seed: {torch.initial_seed()}")

# #     pipeline(
# #         video_path=args.video_path,
# #         audio_path=args.audio_path,
# #         video_out_path=args.video_out_path,
# #         video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
# #         num_frames=config.data.num_frames,
# #         num_inference_steps=args.inference_steps,
# #         guidance_scale=args.guidance_scale,
# #         weight_dtype=dtype,
# #         width=config.data.resolution,
# #         height=config.data.resolution,
# #     )


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
# #     parser.add_argument("--inference_ckpt_path", type=str, required=True)
# #     parser.add_argument("--video_path", type=str, required=True)
# #     parser.add_argument("--audio_path", type=str, required=True)
# #     parser.add_argument("--video_out_path", type=str, required=True)
# #     parser.add_argument("--inference_steps", type=int, default=20)
# #     parser.add_argument("--guidance_scale", type=float, default=1.0)
# #     parser.add_argument("--seed", type=int, default=1247)
# #     args = parser.parse_args()

# #     config = OmegaConf.load(args.unet_config_path)

# #     main(config, args)



# import argparse
# from gfpgan import GFPGAN
# from codeformer import CodeFormer

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--superres', choices=['GFPGAN', 'CodeFormer', 'both'], help='Super-resolution method')
#     return parser.parse_args()

# # def apply_super_resolution(image, method):
# #     if method == 'GFPGAN':
# #         # Apply GFPGAN super-resolution
# #         pass
# #     elif method == 'CodeFormer':
# #         # Apply CodeFormer super-resolution
# #         pass
# #     elif method == 'both':
# #         # Apply both GFPGAN and CodeFormer super-resolution
# #         pass
# #     return image


# def apply_super_resolution(image, method):
#     if method == 'GFPGAN':
#         # Initialize GFPGAN model
#         gfpgan_model = GFPGAN()
#         # Apply GFPGAN super-resolution
#         enhanced_image = gfpgan_model.enhance(image)
#     elif method == 'CodeFormer':
#         # Initialize CodeFormer model
#         codeformer_model = CodeFormer()
#         # Apply CodeFormer super-resolution
#         enhanced_image = codeformer_model.enhance(image)
#     elif method == 'both':
#         # Apply both GFPGAN and CodeFormer super-resolution
#         enhanced_image = apply_super_resolution(image, 'GFPGAN')
#         enhanced_image = apply_super_resolution(enhanced_image, 'CodeFormer')
#     return enhanced_image


# def main():
#     args = parse_args()
#     # Load input frame
#     input_frame = load_image('input_frame.jpg')
#     # Generate lipsynced frame
#     lipsynced_frame = generate_lipsynced_frame(input_frame)
#     # Apply super-resolution if specified
#     if args.superres:
#         lipsynced_frame = apply_super_resolution(lipsynced_frame, args.superres)
#     # Save or display the enhanced frame
#     save_image(lipsynced_frame, 'enhanced_frame.jpg')

# if __name__ == '__main__':
#     main()

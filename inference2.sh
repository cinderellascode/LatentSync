# inference.sh
python inference.py \
    --input_video "$INPUT_VIDEO" \
    --output_video "$OUTPUT_VIDEO" \
    --superres "$SUPERRES" 


#!/bin/bash

# python inference.py \
#     --unet_config_path configs/unet.yaml \
#     --inference_ckpt_path checkpoints/latentsync.pth \
#     --video_path input_video.mp4 \
#     --audio_path input_audio.wav \
#     --video_out_path output_video.mp4 \
#     --superres GFPGAN  # or CodeFormer


#!/bin/bash

# python -m scripts.inference \
#     --unet_config_path "configs/unet/second_stage.yaml" \
#     --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
#     --inference_steps 20 \
#     --guidance_scale 1.5 \
#     --video_path "assets/demo1_video.mp4" \
#     --audio_path "assets/demo1_audio.wav" \
#     --video_out_path "video_out.mp4"


# SUPERRES=$1

# python inference.py --superres $SUPERRES

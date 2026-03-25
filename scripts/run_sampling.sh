#/bin/bash

python3 sample_condition_final.py \
    --model_config=configs/model_imagenet_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/inpainting_imagenet_config.yaml \
    --gpu=2 \
    --save_dir=./results_saliency_step_final_Mar_SNR_-5_highest \
    --c_rate=0.95 \
    --particle_size=5 \
    --timestep_respacing=200 \
    --u2net_model_path=/home/mxxie/SemAID/U_2_Net/model/u2net.pth \
    --metrics_file=image_metrics.txt \
    --snr_db=-5.0
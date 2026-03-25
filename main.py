import os
import cv2

from functools import partial
import argparse
import yaml
import time

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from piq import psnr, ssim
from piq.perceptual import LPIPS
from PIL import Image
import timm
import json
import torchvision.models as models
import torch.nn as nn
from skimage.feature import local_binary_pattern

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.svd_replacement import Deblurring, Deblurring2D
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from U_2_Net.u2net_saliency import U2NetSaliencyDetector

import math
from scipy.stats import entropy

from util.semantic_utils import load_inception_model, calculate_feature_complexity, combine_with_traditional_features, calculate_entropy, nonlinear_entropy_to_mask_prob
from util.semantic_utils import save_image_metrics, save_time_summary, calculate_noise_variance_from_snr

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./results_saliency_step_final_Mar_SNR_-5_highest')
    parser.add_argument('--c_rate', type=float, default=0.99)
    parser.add_argument('--particle_size', type=int, default=5)
    parser.add_argument('--timestep_respacing', type=str, default='200',help='use spaced steps for faster sampling (e.g. 50 or ddim25)')
    parser.add_argument('--u2net_model_path', type=str, default='/home/mxxie/SemAID/U_2_Net/model/u2net.pth',help='Path to U2Net pretrained model')
    parser.add_argument('--metrics_file', type=str, default='image_metrics.txt', help='Text file to save image metrics')
    parser.add_argument('--snr_db', type=float, default=-5.0, help='Signal-to-noise ratio in dB')   
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    saliency_detector = U2NetSaliencyDetector(
        model_path=args.u2net_model_path, 
        model_type='u2net', 
        device=device
    )  
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    inception_model, inception_transform = load_inception_model(device)
    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    original_noise_config = measure_config['noise'].copy()
    logger.info(f"Operation: {measure_config['operator']['name']} / Original noise config: {original_noise_config}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, None, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config, c_rate=args.c_rate, particle_size=args.particle_size) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    batch_size = 1  # Do not change this value. Larger batch size is not available for particle size > 1.
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.CenterCrop((256, 256)),
                                    transforms.Resize((256, 256)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Preprocessing shared by FFHQ and ImageNet.
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=0, train=False)

    metrics_file_path = os.path.join(out_path, args.metrics_file)

    total_time = 0.0
    time_list = []
    # (Exception) In case of inpainting, we need to generate a mask 
  # if measure_config['operator']['name'] == 'inpainting':
   #    mask_gen = mask_generator(
    #      **measure_config['mask_opt']
     #  )
    # Do inference
    
    for i, ref_img in enumerate(loader):

        logger.info(f"Inference for image {i}")
        fnames = [str(j).zfill(5) + '.png' for j in range(i * batch_size, (i+1) * batch_size)]
        ref_img = ref_img.to(device)
        # Record start time
        start_time = time.time()

        if measure_config['operator'] ['name'] == 'inpainting':

    # Compute complexity using Inception features    
            complexity_score, complexity_metrics = calculate_feature_complexity(
                ref_img[0], inception_model, inception_transform, device
            )
            logger.info(f"Image {i} complexity analysis:")

            logger.info(f"Hybrid complexity score: {complexity_score:.4f}")
            logger.info(f"Deep learning complexity: {complexity_metrics['dl_complexity']:.4f}")
            
            ref_img_np = ref_img[0].permute(1, 2, 0).detach().cpu().numpy()
            ref_img_np = (ref_img_np + 1) * 127.5
            ref_img_np = ref_img_np.astype(np.uint8)

    # Canny edge detection
            gray = cv2.cvtColor(ref_img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 250, 350)
            edge_ratio = np.mean(edges > 0)

    # Convert edge map to tensor for edge protection
            edge_tensor = torch.from_numpy(edges > 0).float().to(device)

    # Obtain saliency map using U2Net
            ref_img_pil = Image.fromarray(ref_img_np)
            saliency_map = saliency_detector.predict_saliency_from_image(ref_img_pil)
            saliency_map = cv2.resize(saliency_map, (256, 256), interpolation=cv2.INTER_LINEAR)
            saliency_tensor = torch.from_numpy(saliency_map).float().to(device)

    # Get dynamic adjustment parameters from config
            mask_config = measure_config['mask_opt']
            base_ratio = mask_config['base_mask_ratio']
            adjust_strength = mask_config['edge_adjust_strength']
    
    # Compute adjusted mask ratio
            if measure_config['mask_opt'] ['dynamic_mask'] == 'True':
                adjusted_ratio = nonlinear_entropy_to_mask_prob(complexity_score,min_mask_prob=0.89,max_mask_prob=0.94,min_entropy=0.19,max_entropy=0.23,sensitivity=2.0)
            else :
                adjusted_ratio = nonlinear_entropy_to_mask_prob(complexity_score,min_mask_prob=0.89,max_mask_prob=0.94,min_entropy=0.19,max_entropy=0.23,sensitivity=2.0)
                 
    # Visualization save (added edge and mask images)
            plt.imsave(os.path.join(out_path, 'progress', f'{i}_original.png'), cv2.cvtColor(ref_img_np, cv2.COLOR_RGB2BGR))
            plt.imsave(os.path.join(out_path, 'progress', f'{i}_edges.png'), edges, cmap='gray')
            plt.imsave(os.path.join(out_path, 'progress', f'{i}_saliency.png'), saliency_map, cmap='hot')
    # Generate dynamic mask   
            dynamic_mask_opt = {
                'mask_type': mask_config['mask_type'],
                'mask_prob_range': [adjusted_ratio, adjusted_ratio+0.01],  # Keep original range format
                'image_size': mask_config['image_size'],
                'edge_aware': mask_config['edge_aware'],
                'edge_strength': mask_config['edge_strength'],
                'adjusted_ratio': adjusted_ratio
            }
            # Masks only exist in the inpainting tasks. 
            mask_gen = mask_generator(**dynamic_mask_opt)
            mask = mask_gen(ref_img,semantic_importance=saliency_tensor.unsqueeze(0))
            mask = mask[0, 0, :, :].unsqueeze(dim=0).unsqueeze(dim=0)

            mask_prob = 1 - mask.mean().item()  # since mask=1 indicates keep, 0 indicates mask, so use 1 minus

            plt.imsave(os.path.join(out_path, 'progress', f'{i}_mask.png'), mask[0,0].cpu().numpy(), cmap='gray')

            if 0.9<mask_prob<0.99:
                steps_adjust=1000
            elif 0.8<mask_prob<0.98:
                steps_adjust=500
            elif 0.7<mask_prob<0.8:
                steps_adjust=250
            elif 0.6<mask_prob<0.7:
                steps_adjust=100
            else:
                steps_adjust=100
            diffusion_config['timestep_respacing'] = steps_adjust

            logger.info(f"""
            ======== Image {i} Analysis Results ========
            Original mask ratio: {base_ratio:.4f}
            Edge pixel ratio: {edge_ratio:.4f}
            Feature complexity score: {complexity_score:.4f}
            Final mask ratio for subject: {adjusted_ratio:.4f}
            Overall average mask: {mask_prob:.4f}
            Denoising steps: {steps_adjust}
            ===========================================
            """)

            # save_image_metrics(i, mask_prob, steps_adjust, adjusted_ratio, 
            #                   complexity_score, edge_ratio, metrics_file_path)

            sampler = create_sampler(**diffusion_config, c_rate=args.c_rate, particle_size=args.particle_size) 
            sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
            # measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            # sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator = operator, op = 'inpainting', mask = mask)

            y = operator.forward(ref_img, mask=mask)
            noise_variance = calculate_noise_variance_from_snr(y, args.snr_db, mask=mask)
            measure_config['noise'] = original_noise_config.copy()
            measure_config['noise']['sigma'] = math.sqrt(noise_variance)
            
            # Create noiser with updated noise config
            noiser = get_noise(**measure_config['noise'])
            logger.info(f"Noise variance computed from SNR {args.snr_db}dB: {noise_variance:.6f}")

            cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator=operator, op='inpainting', mask=mask)

            y_n = noiser(y)
         
        # Sampling
        # If you wish to record the intermediate steps, turn record = True below.
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path).requires_grad_()

        # Record end time
        end_time = time.time()
        time_taken = end_time - start_time
        total_time += time_taken
        time_list.append(time_taken)
        save_image_metrics(i, mask_prob, steps_adjust, adjusted_ratio, 
                          complexity_score, edge_ratio, metrics_file_path)
        logger.info(f"Image {i} generation time: {time_taken:.2f} seconds")

        for _ in range(batch_size):
            plt.imsave(os.path.join(out_path, 'input', fnames[_]), clear_color(y_n[_,:,:,:].unsqueeze(dim=0)))
            plt.imsave(os.path.join(out_path, 'label', fnames[_]), clear_color(ref_img[_,:,:,:].unsqueeze(dim=0)))
            plt.imsave(os.path.join(out_path, 'recon', fnames[_]), clear_color(sample[_,:,:,:].unsqueeze(dim=0)))

    if time_list:
        avg_time = total_time / len(time_list)
        logger.info(f"All images processed, total time: {total_time:.2f} seconds")
        logger.info(f"Average generation time per image: {avg_time:.2f} seconds")
        logger.info(f"Average generation time per image: {avg_time/60:.2f} minutes")
        
        # Save time summary to file
        save_time_summary(total_time, avg_time, len(time_list), metrics_file_path)
        
if __name__ == '__main__':
    main()
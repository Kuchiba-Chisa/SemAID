import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
import yaml
import torch.nn as nn

def load_inception_model(device):
    inception_model = models.inception_v3(pretrained=True)
    inception_model.eval()
    inception_model.to(device)
    
    # Remove the final classification layer to obtain feature extractor
    inception_model.fc = nn.Identity()
    
    # Store intermediate layer features
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hooks to extract features from multiple intermediate layers
    inception_model.Mixed_5b.register_forward_hook(get_activation('Mixed_5b'))
    inception_model.Mixed_5c.register_forward_hook(get_activation('Mixed_5c'))
    inception_model.Mixed_5d.register_forward_hook(get_activation('Mixed_5d'))
    inception_model.Mixed_6a.register_forward_hook(get_activation('Mixed_6a'))
    inception_model.Mixed_6b.register_forward_hook(get_activation('Mixed_6b'))
    inception_model.Mixed_6c.register_forward_hook(get_activation('Mixed_6c'))
    inception_model.Mixed_6d.register_forward_hook(get_activation('Mixed_6d'))
    inception_model.Mixed_6e.register_forward_hook(get_activation('Mixed_6e'))
    inception_model.Mixed_7a.register_forward_hook(get_activation('Mixed_7a'))
    inception_model.Mixed_7b.register_forward_hook(get_activation('Mixed_7b'))
    inception_model.Mixed_7c.register_forward_hook(get_activation('Mixed_7c'))
    
    return inception_model, activation

# Compute image complexity using Inception features
def calculate_feature_complexity(image_tensor, inception_model, activation, device):
    # Preprocessing for InceptionV3
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Ensure input image is in [0,1] range
    if image_tensor.min() < 0:
        image_tensor = (image_tensor + 1) / 2.0
    
    # Preprocess image
    input_tensor = preprocess(image_tensor)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        _ = inception_model(input_batch)
    
    # Compute complexity for multiple layers
    layer_complexities = []
    layer_stats = []
    
    # Weights for different layers (deeper layers have higher weight)
    layer_weights = {
        'Mixed_5b': 0.1, 'Mixed_5c': 0.1, 'Mixed_5d': 0.1,
        'Mixed_6a': 0.2, 'Mixed_6b': 0.2, 'Mixed_6c': 0.2,
        'Mixed_6d': 0.3, 'Mixed_6e': 0.3,
        'Mixed_7a': 0.4, 'Mixed_7b': 0.4, 'Mixed_7c': 0.4
    }
    
    for layer_name, features in activation.items():
        if layer_name not in layer_weights:
            continue
            
        # Compute statistics for this layer
        flat_features = features.view(features.size(0), -1)
        
        # 1. Activation entropy
        activation_entropy = 0
        for i in range(flat_features.size(0)):
            probs = torch.softmax(flat_features[i], dim=0)
            e = entropy(probs.cpu().numpy())
            activation_entropy += e
        activation_entropy /= flat_features.size(0)
        
        # 2. Activation variance
        activation_variance = torch.var(features).item()
        
        # 3. Sparsity (Gini coefficient)
        sorted_activations, _ = torch.sort(flat_features.abs(), dim=1, descending=True)
        cumsum = torch.cumsum(sorted_activations, dim=1)
        gini_coefficient = 1 - 2 * torch.mean(cumsum / cumsum[:, -1:].clamp(min=1e-10))
        
        # 4. Activation kurtosis (measures peakiness of distribution)
        activation_kurtosis = torch.mean((features - features.mean())**4) / (features.std()**4 + 1e-10)
        
        # 5. Non-zero activation ratio
        nonzero_ratio = (features.abs() > 0.01).float().mean().item()
        
        # Normalize these metrics (based on empirical values)
        norm_entropy = np.clip(activation_entropy / 8.0, 0, 1)
        norm_variance = np.clip(activation_variance / 2.0, 0, 1)
        norm_gini = np.clip(gini_coefficient.item() / 0.8, 0, 1)
        norm_kurtosis = np.clip(activation_kurtosis.item() / 10.0, 0, 1)
        norm_nonzero = np.clip(nonzero_ratio / 0.5, 0, 1)
        
        # Combine metrics
        layer_complexity = (
            0.3 * norm_entropy + 
            0.2 * norm_variance + 
            0.2 * norm_gini + 
            0.15 * norm_kurtosis + 
            0.15 * norm_nonzero
        ) * layer_weights[layer_name]
        
        layer_complexities.append(layer_complexity)
        layer_stats.append({
            'layer': layer_name,
            'entropy': activation_entropy,
            'variance': activation_variance,
            'gini': gini_coefficient.item(),
            'kurtosis': activation_kurtosis.item(),
            'nonzero_ratio': nonzero_ratio,
            'complexity': layer_complexity
        })
    
    # Compute overall complexity score (weighted average)
    total_complexity = np.mean(layer_complexities)
    
    # Combine with traditional image features
    hybrid_complexity = combine_with_traditional_features(image_tensor, total_complexity)
    
    return hybrid_complexity, {
        'hybrid_complexity': hybrid_complexity,
        'dl_complexity': total_complexity,
        'layer_stats': layer_stats,
        'num_layers': len(layer_complexities)
    }

def combine_with_traditional_features(image_tensor, dl_complexity):
    # Convert to numpy image for traditional feature extraction
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # Compute traditional image complexity features
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 1. Edge complexity
    edges = cv2.Canny(gray_image, 100, 200)
    edge_density = np.mean(edges > 0)
    
    # 2. Color complexity
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    color_std = np.std(hsv_image, axis=(0, 1))
    color_complexity = np.mean(color_std) / 64.0  # normalize
    
    # 3. Texture complexity (using LBP)
    lbp = local_binary_pattern(gray_image, 8, 1, method='uniform')
    texture_complexity = len(np.unique(lbp)) / 256.0  # normalize
    
    # 4. Information entropy
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    image_entropy = -np.sum(hist * np.log2(hist + 1e-10))
    norm_entropy = image_entropy / 8.0  # normalize
    
    # Combine traditional features
    traditional_complexity = (
        0.3 * edge_density + 
        0.2 * color_complexity + 
        0.3 * texture_complexity + 
        0.2 * norm_entropy
    )
    
    # Combine deep learning and traditional features
    hybrid_complexity = 0.7 * dl_complexity + 0.3 * traditional_complexity
    
    return hybrid_complexity

def calculate_entropy(image_np):
    """
    Calculate information entropy of an image.
    """
    # Convert to grayscale
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # RGB image
        gray_image = 0.2989 * image_np[:,:,0] + 0.5870 * image_np[:,:,1] + 0.1140 * image_np[:,:,2]
    else:  # already grayscale
        gray_image = image_np
    
    # Quantize to 256 gray levels
    quantized_image = (gray_image).astype(np.uint8)
    
    # Compute histogram
    hist, _ = np.histogram(quantized_image.flatten(), bins=256, range=(0, 255))
    
    # Compute probability distribution
    prob_dist = hist / float(np.sum(hist))
    
    # Compute information entropy
    img_entropy = entropy(prob_dist, base=2)
    
    normalized_entropy = img_entropy / 8.0

    return img_entropy, normalized_entropy

def nonlinear_entropy_to_mask_prob(normalized_entropy, 
                                  min_mask_prob=0.75, 
                                  max_mask_prob=0.92,
                                  min_entropy=0.15,
                                  max_entropy=0.25,
                                  sensitivity=2.0):
    """
    Map normalized information entropy to mask probability using a non-linear function.
    
    Args:
        normalized_entropy: normalized information entropy (0.15-0.25)
        min_mask_prob: minimum mask probability (default 0.6)
        max_mask_prob: maximum mask probability (default 0.9)
        min_entropy: minimum entropy value (default 0.15)
        max_entropy: maximum entropy value (default 0.25)
        sensitivity: sensitivity parameter, steeper slope in the middle region with higher value
    """
    # Clamp entropy to reasonable range
    clamped_entropy = max(min_entropy, min(normalized_entropy, max_entropy))
    
    # Normalize to [0,1] range
    normalized = (clamped_entropy - min_entropy) / (max_entropy - min_entropy)
    
    # Use sigmoid function to create non-linear mapping
    from scipy.special import expit
    # Adjust sensitivity
    adjusted = expit(sensitivity * (normalized - 0.5) * 6)  # scaling factor 6 makes sigmoid steeper
    
    # Map to mask probability range
    mask_prob = max_mask_prob - (max_mask_prob - min_mask_prob) * adjusted
    
    return mask_prob


def save_image_metrics(image_idx, mask_prob, denoise_steps, adjusted_ratio, 
                      normalized_entropy, edge_ratio, save_path):
    """
    Save image metrics to a text file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # If file does not exist, create it and write header
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            f.write("image_idx\tmask_prob\tdenoise_steps\tadjusted_ratio\tnormalized_entropy\tedge_ratio\n")
    
    # Append current image metrics
    with open(save_path, 'a') as f:
        f.write(f"{image_idx}\t{mask_prob:.4f}\t{denoise_steps}\t{adjusted_ratio:.4f}\t{normalized_entropy:.4f}\t{edge_ratio:.4f}\n")

def save_time_summary(total_time, avg_time, num_images, save_path):
    """
    Save time summary information.
    """
    with open(save_path, 'a') as f:
        f.write(f"\n\n=== Time Summary ===\n")
        f.write(f"Total images processed: {num_images}\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        f.write(f"Average time per image: {avg_time:.2f} seconds\n")
        f.write(f"Average time per image: {avg_time/60:.2f} minutes\n")

def calculate_noise_variance_from_snr(y, snr_db, mask=None):
    """
    Calculate noise variance based on given SNR.
    
    Args:
        y: input image tensor
        snr_db: signal-to-noise ratio in dB
        mask: optional mask for computing signal power only on unmasked regions
    
    Returns:
        noise_variance: noise variance
    """
    # Compute signal power
    if mask is not None:
        # If mask is provided, only consider unmasked regions
        signal = y * mask
        signal_power = torch.mean(signal**2)
    else:
        signal_power = torch.mean(y**2)
    
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10.0)
    
    # Compute required noise power
    noise_power = signal_power / snr_linear
    
    return noise_power.item()

import numpy as np
import torch
import scipy
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from .fastmri_utils import fft2c_new, ifft2c_new
import cv2
from U_2_Net.u2net_saliency import U2NetSaliencyDetector

"""
Helper functions for new types of inverse problems
"""

def fft2(x):
    """FFT with shifting DC to the center of the image"""
    return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
    """IFFT with shifting DC to the corner of the image prior to transform"""
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


def fft2_m(x):
    """FFT for multi-coil"""
    if not torch.is_complex(x):
        x = x.type(torch.complex64)
    return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
    """IFFT for multi-coil"""
    if not torch.is_complex(x):
        x = x.type(torch.complex64)
    return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(x)


def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))


def normalize_np(img):
    """Normalize img in arbitrary range to [0, 1]"""
    img -= np.min(img)
    img /= np.max(img)
    return img


def prepare_im(load_dir, image_size, device):
    ref_img = torch.from_numpy(normalize_np(plt.imread(load_dir)[:, :, :3].astype(np.float32))).to(device)
    ref_img = ref_img.permute(2, 0, 1)
    ref_img = ref_img.view(1, 3, image_size, image_size)
    ref_img = ref_img * 2 - 1
    return ref_img


def fold_unfold(img_t, kernel, stride):
    img_shape = img_t.shape
    B, C, H, W = img_shape
    print("\n----- input shape: ", img_shape)

    patches = img_t.unfold(3, kernel, stride).unfold(2, kernel, stride).permute(0, 1, 2, 3, 5, 4)

    print("\n----- patches shape:", patches.shape)
    patches = patches.contiguous().view(B, C, -1, kernel*kernel)
    print("\n", patches.shape)
    patches = patches.permute(0, 1, 3, 2)
    print("\n", patches.shape)
    patches = patches.contiguous().view(B, C*kernel*kernel, -1)
    print("\n", patches.shape)

    output = F.fold(patches, output_size=(H, W),
                    kernel_size=kernel, stride=stride)
    recovery_mask = F.fold(torch.ones_like(patches), output_size=(
        H, W), kernel_size=kernel, stride=stride)
    output = output/recovery_mask

    return patches, output


def reshape_patch(x, crop_size=128, dim_size=3):
    x = x.transpose(0, 2).squeeze()
    x = x.view(dim_size**2, 3, crop_size, crop_size)
    return x


def reshape_patch_back(x, crop_size=128, dim_size=3):
    x = x.view(dim_size**2, 3*(crop_size**2)).unsqueeze(dim=-1)
    x = x.transpose(0, 2)
    return x


class Unfolder:
    def __init__(self, img_size=256, crop_size=128, stride=64):
        self.img_size = img_size
        self.crop_size = crop_size
        self.stride = stride

        self.unfold = nn.Unfold(crop_size, stride=stride)
        self.dim_size = (img_size - crop_size) // stride + 1

    def __call__(self, x):
        patch1D = self.unfold(x)
        patch2D = reshape_patch(patch1D, crop_size=self.crop_size, dim_size=self.dim_size)
        return patch2D


def center_crop(img, new_width=None, new_height=None):
    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


class Folder:
    def __init__(self, img_size=256, crop_size=128, stride=64):
        self.img_size = img_size
        self.crop_size = crop_size
        self.stride = stride

        self.fold = nn.Fold(img_size, crop_size, stride=stride)
        self.dim_size = (img_size - crop_size) // stride + 1

    def __call__(self, patch2D):
        patch1D = reshape_patch_back(patch2D, crop_size=self.crop_size, dim_size=self.dim_size)
        return self.fold(patch1D)


def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random square mask for inpainting"""
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=224, margin=(16, 16), edge_aware=False, edge_strength=0.3, adjusted_ratio=0.5):
        """
        mask_type: one of ['box', 'random', 'both', 'extreme', 'edge_aware', 'semantic']
        mask_len_range: range of box size (min, max)
        mask_prob_range: range of pixel masking probability (min, max) for random masking
        image_size: size of the image (assumed square)
        margin: margin for box placement (height, width)
        edge_aware: whether to apply edge protection
        edge_strength: strength of edge protection (unused in current implementation)
        adjusted_ratio: ratio for adjusting mask probability in semantic masking
        """
        assert mask_type in ['box', 'random', 'both', 'extreme', 'edge_aware', 'semantic']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin
        self.edge_aware = edge_aware
        self.edge_strength = edge_strength
        self.adjusted_ratio = adjusted_ratio

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def _apply_edge_protection(self, base_mask, img):
        """Adjust base mask to protect edges detected by Canny"""
        img_np = img[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np + 1) * 127.5  # convert from [-1,1] to [0,255]
        gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=250, threshold2=350)
        edge_ratio = np.mean(edges > 0)
        if edge_ratio < 0.1:
            edge_mask = torch.from_numpy(edges > 0).to(
                device=img.device,
                dtype=torch.bool
            ).unsqueeze(0).unsqueeze(0)
            protected_mask = base_mask.clone()
            protected_mask[..., edge_mask.squeeze()] = 1.0
            return protected_mask
        else:
            return base_mask

    def _retrieve_semantic(self, img, semantic_importance):
        """Generate mask based on semantic importance map"""
        assert semantic_importance.shape[-2:] == (self.image_size, self.image_size),\
            f"Semantic map shape {semantic_importance.shape} mismatch with image size {self.image_size}"

        semantic_importance = (semantic_importance - semantic_importance.min()) /\
                            (semantic_importance.max() - semantic_importance.min() + 1e-6)

        saliency_threshold = 0.5
        high_saliency_mask = (semantic_importance > saliency_threshold).float()
        low_saliency_mask = 1 - high_saliency_mask

        adjusted_ratio = self.adjusted_ratio

        high_rand_matrix = torch.rand_like(semantic_importance)
        high_saliency_area_mask = (high_rand_matrix > adjusted_ratio).float() * high_saliency_mask

        low_rand_matrix = torch.rand_like(semantic_importance)
        low_saliency_area_mask = (low_rand_matrix > 0.96).float() * low_saliency_mask

        base_mask = high_saliency_area_mask + low_saliency_area_mask

        if base_mask.dim() == 2:
            base_mask = base_mask.unsqueeze(0)
        base_mask = base_mask.repeat(3, 1, 1)

        return base_mask.unsqueeze(0).to(img.device)

    def __call__(self, img, semantic_importance=None):
        if self.mask_type == 'random':
            base_mask = self._retrieve_random(img)
        elif self.mask_type == 'box':
            base_mask, t, th, w, wl = self._retrieve_box(img)
        elif self.mask_type == 'extreme':
            base_mask, t, th, w, wl = self._retrieve_box(img)
            base_mask = 1. - base_mask
        elif self.mask_type == 'edge_aware':
            base_mask = self._retrieve_random(img)
        elif self.mask_type == 'semantic':
            assert semantic_importance is not None, "semantic_importance must be provided for semantic masking"
            base_mask = self._retrieve_semantic(img, semantic_importance)
        else:
            raise NotImplementedError
        if self.edge_aware:
            return self._apply_edge_protection(base_mask, img)
        return base_mask


def unnormalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img / scaling


def normalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img * scaling


def dynamic_thresholding(img, s=0.95):
    img = normalize(img, s=s)
    return torch.clip(img, -1., 1.)


def get_gaussian_kernel(kernel_size=31, std=0.5):
    n = np.zeros([kernel_size, kernel_size])
    n[kernel_size//2, kernel_size//2] = 1
    k = scipy.ndimage.gaussian_filter(n, sigma=std)
    k = k.astype(np.float32)
    return k


def init_kernel_torch(kernel, device="cuda:0"):
    h, w = kernel.shape
    kernel = Variable(torch.from_numpy(kernel).to(device), requires_grad=True)
    kernel = kernel.view(1, 1, h, w)
    kernel = kernel.repeat(1, 3, 1, 1)
    return kernel


class exact_posterior():
    def __init__(self, betas, sigma_0, label_dim, input_dim):
        self.betas = betas
        self.sigma_0 = sigma_0
        self.label_dim = label_dim
        self.input_dim = input_dim

    def py_given_x0(self, x0, y, A, verbose=False):
        norm_const = 1/((2 * np.pi)**self.input_dim * self.sigma_0**2)
        exp_in = -1/(2 * self.sigma_0**2) * torch.linalg.norm(y - A(x0))**2
        if not verbose:
            return norm_const * torch.exp(exp_in)
        else:
            return norm_const * torch.exp(exp_in), norm_const, exp_in

    def pxt_given_x0(self, x0, xt, t, verbose=False):
        beta_t = self.betas[t]
        norm_const = 1/((2 * np.pi)**self.label_dim * beta_t)
        exp_in = -1/(2 * beta_t) * torch.linalg.norm(xt - np.sqrt(1 - beta_t)*x0)**2
        if not verbose:
            return norm_const * torch.exp(exp_in)
        else:
            return norm_const * torch.exp(exp_in), norm_const, exp_in

    def prod_logsumexp(self, x0, xt, y, A, t):
        py_given_x0_density, pyx0_nc, pyx0_ei = self.py_given_x0(x0, y, A, verbose=True)
        pxt_given_x0_density, pxtx0_nc, pxtx0_ei = self.pxt_given_x0(x0, xt, t, verbose=True)
        summand = (pyx0_nc * pxtx0_nc) * torch.exp(-pxtx0_ei - pxtx0_ei)
        return torch.logsumexp(summand, dim=0)


def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def total_variation_loss(img, weight):
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2)).mean()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2)).mean()
    return weight * (tv_h + tv_w)


if __name__ == '__main__':
    import numpy as np
    from torch import nn
    import matplotlib.pyplot as plt
    device = 'cuda:0'
    load_path = '/media/harry/tomo/FFHQ/256/test/00000.png'
    img = torch.tensor(plt.imread(load_path)[:, :, :3])
    img = torch.permute(img, (2, 0, 1)).view(1, 3, 256, 256).to(device)

    mask_len_range = (32, 128)
    mask_prob_range = (0.3, 0.7)
    image_size = 256
    mask_gen = mask_generator(
        mask_len_range=mask_len_range,
        mask_prob_range=mask_prob_range,
        image_size=image_size
    )
    mask = mask_gen(img)

    mask = np.transpose(mask.squeeze().cpu().detach().numpy(), (1, 2, 0))

    plt.imshow(mask)
    plt.show()
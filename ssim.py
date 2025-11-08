# This file provides the L1/SSIM loss function and metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

# --- Metrics ---
def psnr(target, prediction, max_val=None):
    # [cite: 155]
    if max_val is None:
        max_val = target.max()
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(max_val) - 10 * torch.log10(mse)

# --- SSIM Implementation ---
# [cite: 156]
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window.data
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.type())
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# --- Combined Loss Function ---
class CombinedLoss(nn.Module):
    """
    Implements the combined L1 and SSIM loss: L = lambda1 * L1 + lambda2 * (1 - SSIM) 
    """
    def __init__(self, lambda1=1.0, lambda2=1.0, window_size=11):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(window_size=window_size, size_average=True)

    def forward(self, prediction, target):
        l1 = self.l1_loss(prediction, target)
        ssim = self.ssim_loss(prediction, target)
        loss = (self.lambda1 * l1) + (self.lambda2 * (1.0 - ssim)) # 
        return loss

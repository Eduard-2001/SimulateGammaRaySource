import torch
import torch.nn.functional as F

def gaussian(size, sigma):
    x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    x, y = torch.meshgrid(x, y, indexing='ij')  # 生成网格
    g = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def convolve2d(img, kernel):
    # 使用F.conv2d进行卷积
    return F.conv2d(img, kernel.unsqueeze(0).unsqueeze(0), padding=kernel.shape[0]//2)

def ssim_function(img1, img2, window_size=11, data_range=255.0, sigma=1.5):
    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 创建高斯窗口
    window = gaussian(window_size, sigma).to(img1.device)

    # 计算均值
    mu1 = convolve2d(img1, window)
    mu2 = convolve2d(img2, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(img1 ** 2, window) - mu1_sq
    sigma2_sq = convolve2d(img2 ** 2, window) - mu2_sq
    sigma12 = convolve2d(img1 * img2, window) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    return 1-torch.mean(ssim_map)

# 使用示例
# img1和img2需要是torch张量，并且在CUDA设备上
# loss = ssim_function(img1, img2)
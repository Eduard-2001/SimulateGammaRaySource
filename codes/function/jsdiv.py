import torch
import torch.nn.functional as F

def jsdiv(img1, img2):
    img1 = img1.unsqueeze(0).unsqueeze(0)
    img2 = img2.unsqueeze(0).unsqueeze(0)
    img1 = img1/img1.sum()
    img2 = img2/img2.sum() # normalized to 1, as js divergence applies for probability distributions. 
                            # img now could be understood as the probability distribution of a single photon in the sky.
    ks12 = torch.sum(img1*torch.log(img1/img2))
    ks21 = torch.sum(img2*torch.log(img2/img1))
    return 1-(ks12+ks21)/2

# 使用示例
# img1和img2需要是torch张量，并且在CUDA设备上
# loss = ssim_function(img1, img2)
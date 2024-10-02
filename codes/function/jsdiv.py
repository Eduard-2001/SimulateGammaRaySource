import torch
import torch.nn.functional as F

def jsdiv(img1, img2):
    #print(img1.shape)
    #img1 = img1.unsqueeze(0).unsqueeze(0)
    #img2 = img2.unsqueeze(0).unsqueeze(0)
    #print(img1.shape,img2.shape)
    #print(img1.max())
    #img1 = img1+ 1e-20 #torch.tensor(1e-20)[torch.newaxis,torch.newaxis,torch.newaxis,torch.newaxis]
    #img2 = img1+ 1e-20#torch.tensor(1e-20)[torch.newaxis,torch.newaxis,torch.newaxis,torch.newaxis]
    #print(img1)
    #img1 = img1/(img1.sum(-1).sum(-1))[:,:,torch.newaxis,torch.newaxis]
    #img2 = img2/(img2.sum(-1).sum(-1))[:,:,torch.newaxis,torch.newaxis] 
    # normalized to 1, as js divergence applies for probability distributions. 
    # img now could be understood as the probability distribution of a single photon in the sky.
    #print(img1.shape)
    #print(img2[0])
    shape = img2.shape
    img1 = img1.reshape(shape[0],shape[1],-1)
    img2 = img2.reshape(shape[0],shape[1],-1)
    img1 = F.softmax(img1,-1)
    img2 = F.softmax(img2,-1)
    img1 = img1.reshape(shape)
    img2 = img2.reshape(shape)
    ks12 = (img1*torch.log(img1/img2)).sum(-1).sum(-1)
    ks21 = (img2*torch.log(img2/img1)).sum(-1).sum(-1)
    #print((img1*torch.log(img1/img2)))
    
    jsdivergence = ((ks12+ks21)/2).mean()
    #print(jsdivergence)
    #raise ValueError('Stop')
    
    return jsdivergence

# 使用示例
# img1和img2需要是torch张量，并且在CUDA设备上
# loss = ssim_function(img1, img2)
EXP_NAME = "EXP_0_1"

# 第一部分

import torch; torch.manual_seed(0)
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from function.Dir import Dir
from function.ssim_torch import ssim_function
from function.jsdiv import jsdiv 
from function.Dataset import ImageDataset
from function.Loss import Custom_criterion
from function.Log import log
import torch.nn.functional as F
import importlib

VAE_module = importlib.import_module(f'function.VAE_{EXP_NAME}')
VAE = getattr(VAE_module, 'VAE')
filepath = "/root/autodl-fs/norandPSF_6.4x6.4.npy"
NUM_TO_LEARN = 2400
NUM_TO_TEST = 400
EPOCHS = 200
BATCH_SIZE = 32 
LATENTDIM = 64
LR_MAX = 5e-4
LR_MIN = 5e-6
mode = 1

DEVICE = 'cuda'
LOSS_PLOT = []
TESTLOSS_PLOT = []
EPOCH_PLOT = []

# 第二部分

name = f'{EPOCHS}epo_{BATCH_SIZE}bth_{LATENTDIM}latn'

dataset = ImageDataset(NUM_TO_LEARN, mode,filepath)
testset = ImageDataset(NUM_TO_TEST,mode,filepath,inverse=True)
dataloader = DataLoader(dataset, BATCH_SIZE, True)
testloader = DataLoader(testset, BATCH_SIZE, True)

vae = VAE(LATENTDIM).to(DEVICE)

lossfunction = jsdiv
optimizer = torch.optim.AdamW(vae.parameters(), lr = LR_MAX)

# 第三部分

def train(dataloader, num_epochs):
    with open(f'training.log', 'w') as nothing: # 清空原log
        pass
    log(f"Experiment name: {EXP_NAME}")
    for epoch in range(num_epochs):
        vae.train() # 切换成训练模式
        total_loss = 0.0
        current_lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * epoch / EPOCHS))
        optimizer = torch.optim.AdamW(vae.parameters(), lr = current_lr)

        for _, (img_LR, img_HR) in enumerate(dataloader):
            img_LR = img_LR.to(DEVICE)
            img_HR = img_HR.to(DEVICE)
            img_SR, _, _ = vae(img_LR)
            img_SR = img_SR.to(DEVICE)
            loss = lossfunction(img_SR,img_HR)
            optimizer.zero_grad()
            loss.backward() # 最耗算力的一步
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader) # 每个EPOCH的loss，全部数据集的平均

        test_loss = 0.0
        for _, (img_LR, img_HR) in enumerate(testloader):
            img_LR = img_LR.to(DEVICE)
            img_HR = img_HR.to(DEVICE)
            img_SR, _, _ = vae(img_LR)
            img_SR = img_SR.to(DEVICE)
            loss = lossfunction(img_SR,img_HR)
            test_loss += loss.item()
        test_avg_loss = test_loss / len(testloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}, Test Loss: {test_avg_loss:.6f}, Current_LR:{current_lr:.4f}")
        log(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}, Test Loss: {test_avg_loss:.6f}, Current_LR:{current_lr:.4f}")

        LOSS_PLOT.append(avg_loss)
        TESTLOSS_PLOT.append(test_avg_loss)
        EPOCH_PLOT.append(epoch)

# 第四部分

datas = np.load(filepath,allow_pickle=True)#.astype(np.object)
blurry_datas = np.stack(datas[:,1])
original_datas = np.stack(datas[:,0])

torch.set_printoptions(precision=10)
lossfunction = jsdiv
print(f'DEVICE:{DEVICE}\n')
print(f'Training Start. \nExperiment name: {EXP_NAME}')
train(dataloader, EPOCHS)
print(f'\n Succsessfully done! Training log saved.')
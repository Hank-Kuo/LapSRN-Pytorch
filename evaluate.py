import argparse
import torch
from torch.autograd import Variable
import torch

parser = argparse.ArgumentParser(description="PyTorch LapSRN Demo")

def psnr(original, compressed): 
    mse = torch.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal .# Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr 

def evaluate(model, data_generator, device):
    avg_psnr1 = 0
    avg_psnr2 = 0
    avg_psnr3 = 0
    total_len = len(data_generator)
    
    for batch in data_generator:
        LR, HR_2_target, HR_4_target, HR_8_target = Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), Variable(batch[4])
        
        LR = LR.to(device)
        HR_2_target = HR_2_target.to(device)
        HR_4_target = HR_4_target.to(device)
        HR_8_target = HR_8_target.to(device)

        HR_2, HR_4, HR_8 = model(LR)
        
        #psnr1 = psnr(HR_2, HR_2_target)
        #psnr2 = psnr(HR_4, HR_4_target)
        #psnr3 = psnr(HR_8, HR_8_target)
        #avg_psnr1 += psnr1
        #avg_psnr2 += psnr2
        #avg_psnr3 += psnr3
    
    return avg_psnr1/total_len, avg_psnr2/total_len, avg_psnr3/total_len

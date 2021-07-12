import argparse
import math

from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser(description='PyTorch LapSRN')

def centeredCrop(img):
    width, height = img.size   # Get dimensions
    new_width = width - width % 8
    new_height = height - height % 8 
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    return img.crop((left, top, right, bottom))

def ycbcr_to_rgb(out, cb, cr):
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img

def generate_bicubic(image, scale):
    LR = image.resize((y.size[0]/8, y.size[1]/8), Image.BICUBIC)
    Bicubic_HR = LR.resize((y.size[0]*sacle, y.size[1]*scale), Image.BICUBIC)
    return Bicubic_HR

def psnr(target, ref, scale):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # assume RGB image
    target_data = np.array(target)
    target_data = target_data[scale:-scale,scale:-scale]
 
    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale,scale:-scale]
 
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(1.0/rmse)

def PSNR(pred, gt, scale=0):
    height, width = pred.shape[:2]
    pred = pred[scale:height - scale, scale:width - scale]
    gt = gt[scale:height - scale, scale:width - scale]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
        
    return 20 * math.log10(255.0 / rmse)

if __name__ == '__init__':
    args = parser.parse_args()
    
    image_path = './data/test/bird.png'
    model_path = './experiment/base_model/checkpoint.tar'
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = torch.load(model_path).to(device)
    model.eval()

    img = Image.open(image_path).convert('YCbCr')
    # img = centeredCrop(img)
    y, cb, cr = img.split()

    LR = y.resize((y.size[0]/8, y.size[1]/8), Image.BICUBIC)
    LR = Variable(ToTensor()(LR)).view(1, -1, LR.size[1], LR.size[0])
    HR_2, HR_4, HR_8 = model(LR)
    HR_2 = HR_2.cpu()
    HR_4 = HR_4.cpu()
    HR_8 = HR_8.cpu()
    
    HR_2 = ycbcr_to_rgb(HR_2, cb, cr)
    HR_4 = ycbcr_to_rgb(HR_4, cb, cr)
    HR_8 = ycbcr_to_rgb(HR_8, cb, cr)
    
    img = img.convert("RGB")
    bi_HR_2 = generate_bicubic(img, 2)
    bi_HR_4 = generate_bicubic(img, 4)
    bi_HR_8 = generate_bicubic(img, 8)


    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0




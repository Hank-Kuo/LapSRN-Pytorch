import argparse
import os
import math

import model.net as net
import model.data_loader as data_loader
import utils.utils as utils

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision.transforms import ToTensor



parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument("--seed", default=1234, help="Seed value.")
parser.add_argument("--dataset_dir", default="./data", help="Path to dataset.")
parser.add_argument("--image_path", default="./data/test/bird.png", help="Path to model checkpoint (by default train from scratch).")
parser.add_argument("--model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")

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
    out_img_y = out[0].data.numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img

def generate_bicubic(LR, scale):
    Bicubic_HR = LR.resize((LR.size[0]*scale, LR.size[1]*scale), Image.BICUBIC)
    return Bicubic_HR


def PSNR(pred, gt, scale=0):
    pred_data = np.array(pred)
    gt_data = np.array(gt)

    height, width = pred_data.shape[:2]
    pred = pred_data[scale:height - scale, scale:width - scale]
    gt = gt_data[scale:height - scale, scale:width - scale]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
        
    return 20 * math.log10(255.0 / rmse)

if __name__ == "__main__":
    args = parser.parse_args()
    output_dir = os.path.join(args.model_dir, 'result')
    params_path = os.path.join(args.model_dir, 'params.json')
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoint')
    
    # params
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = net.Net()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    utils.load_checkpoint(checkpoint_dir, model, optimizer)
    best_model = model.to(params.device)
    model.eval()
    
    img = Image.open(args.image_path).convert('YCbCr')

    LR_image = img.resize((int(img.size[0]/8), int(img.size[1]/8)), Image.BICUBIC)
    LR_x2_image = img.resize((int(img.size[0]/4), int(img.size[1]/4)), Image.BICUBIC)
    LR_x4_image = img.resize((int(img.size[0]/2), int(img.size[1]/2)), Image.BICUBIC)
    LR_x8_image = img.resize((int(img.size[0]/1), int(img.size[1]/1)), Image.BICUBIC)
    # _, cb, cr = LR_x8_image.split()
    y, cb, cr = LR_image.split()
    LR = ToTensor()(y).view(1, -1, y.size[1], y.size[0]).to(params.device)

    HR_2, HR_4, HR_8 = model(LR)

    HR_2 = HR_2.cpu()
    HR_4 = HR_4.cpu()
    HR_8 = HR_8.cpu()
    
    HR_2 = ycbcr_to_rgb(HR_2, cb, cr)
    HR_4 = ycbcr_to_rgb(HR_4, cb, cr)
    HR_8 = ycbcr_to_rgb(HR_8, cb, cr)
    
    LR_image = LR_image.convert("RGB")
    LR_x2_image = LR_x2_image.convert("RGB")
    LR_x4_image = LR_x4_image.convert("RGB")
    LR_x8_image = LR_x8_image.convert("RGB")

    bi_HR_2 = generate_bicubic(LR_image, 2)
    bi_HR_4 = generate_bicubic(LR_image, 4)
    bi_HR_8 = generate_bicubic(LR_image, 8)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    LR_image.save("LR.png")
    LR_x2_image.save("GT_x2.png")
    LR_x4_image.save("GT_x4.png")
    LR_x8_image.save("GT_x8.png")
    bi_HR_2.save("HR_x2_bi.png")
    bi_HR_4.save("HR_x4_bi.png")
    bi_HR_8.save("HR_x8_bi.png")
    HR_2.save("HR_x2.png")
    HR_4.save("HR_x4.png")
    HR_8.save("HR_x8.png")

    psnr_predicted_x2 = PSNR(HR_2, LR_x2_image, scale=2)
    psnr_bicubic_x2 = PSNR(bi_HR_2, LR_x2_image, scale=2)
    psnr_predicted_x4 = PSNR(HR_4, LR_x4_image, scale=4)
    psnr_bicubic_x4 = PSNR(bi_HR_4, LR_x4_image, scale=4)
    psnr_predicted_x8 = PSNR(HR_8, LR_x8_image, scale=8)
    psnr_bicubic_x8 = PSNR(bi_HR_8, LR_x8_image, scale=8)

    fig = plt.figure()
    ax = plt.subplot("131")
    ax.imshow(LR_x8_image, cmap='gray')
    ax.set_title("GT")

    ax = plt.subplot("132")
    ax.imshow(bi_HR_8, cmap='gray')
    ax.set_title("Output(Bicubic) x8")

    ax = plt.subplot("133")
    ax.imshow(HR_8, cmap='gray')
    ax.set_title("Output(LapSRN) x8")
    plt.show()

    
    print('- Eval PSNR Scale 2: Predict: {}, Bicubic: {}'.format(psnr_predicted_x2, psnr_bicubic_x2 ))
    print('- Eval PSNR Scale 4: Predict: {}, Bicubic: {}'.format(psnr_predicted_x4, psnr_bicubic_x4 ))
    print('- Eval PSNR Scale 8: Predict: {}, Bicubic: {}'.format(psnr_predicted_x8, psnr_bicubic_x8 ))


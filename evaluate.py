import argparse

import utils.image as image_utils

import torch
from torch.autograd import Variable
from torchvision import transforms

parser = argparse.ArgumentParser(description="PyTorch LapSRN Demo")
parser.add_argument("--seed", default=1234, help="Seed value.")
parser.add_argument("--image_path", default="./data/test/bird.png", help="Path to model checkpoint (by default train from scratch).")
parser.add_argument("--model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")



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


def evaluate(model, dataset, device, epoch_id):
    avg_psnr1 = 0
    avg_psnr2 = 0
    avg_psnr3 = 0
    
    total_len = len(dataset)
    
    for batch in dataset:
        LR = transforms.ToTensor()(batch[0]).view(1, -1, LR.size[1], LR.size[0]).to(device)
        HR_2, HR_4, HR_8 = model(LR)

        input_image, HR_2_target, HR_4_target, HR_8_target = batch[0], batch[1], batch[2], batch[3]
        input_image = input_image.convert('RGB')
        HR_2_target = HR_2_target.convert('RGB')
        HR_4_target = HR_4_target.convert('RGB')
        HR_8_target = HR_8_target.convert('RGB')

        HR_2 = HR_2.cpu()
        HR_4 = HR_4.cpu()
        HR_8 = HR_8.cpu()
        
        _, cb, cr = batch[3].split()
        HR_2 = image_utils.ycbcr_to_rgb(HR_2, cb, cr)
        HR_4 = image_utils.ycbcr_to_rgb(HR_4, cb, cr)
        HR_8 = image_utils.ycbcr_to_rgb(HR_8, cb, cr)

        psnr1 = PSNR(HR_2, HR_2_target)
        psnr2 = psnr(HR_4, HR_4_target)
        psnr3 = psnr(HR_8, HR_8_target)
        avg_psnr1 += psnr1
        avg_psnr2 += psnr2
        avg_psnr3 += psnr3

        #input_image.save("origin.png")
        #HR_2_target.save("GT_x2.png")
        #HR_4_target.save("GT_x4.png")
        #HR_4_target.save("GT_x8.png")
        #HR_2.save("HR_x2.png")
        #HR_4.save("HR_x4.png")
        #HR_8.save("HR_x8.png")
    
    return avg_psnr1/total_len, avg_psnr2/total_len, avg_psnr3/total_len

if __name__ == "__main__":
    args = parser.parse_args()
    output_dir = os.path.join(args.model_dir, 'result')
    params_path = os.path.join(args.model_dir, 'params.json')
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoint')
    
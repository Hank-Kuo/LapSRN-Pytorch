import os
import argparse
from tqdm import tqdm
import logging

import model.net as net
import model.data_loader as data_loader
import utils.utils as utils
from evaluate import evaluate

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="PyTorch LapSRN")
parser.add_argument("--dataset_path", default="./data", help="Path to dataset.")
parser.add_argument("--model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def main():
    args = parser.parse_args()

     # torch setting
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # os setting
    path = args.dataset_path
    train_dir = os.path.join(path, "train")
    valid_dir = os.path.join(path, "valid")
    params_path = os.path.join(args.model_dir, 'params.json')
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoint')

    # params
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # dataset
    print("===> Loading datasets")
    train_set = data_loader.DatasetFromFolder(train_dir, crop_size=128)
    training_data_loader = DataLoader(dataset=train_set, batch_size=params.batch_size, shuffle=True)

    # Net
    print("===> Building model")
    model = net.Net()
    criterion = net.L1_Charbonnier_loss()
    
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    print(model)

    logging.info("Starting training...")
    for epoch_id in range(1, params.epoch + 1): 
        print("Epoch {}/{}".format(epoch_id, params.epochs))
        model.train()
        running_loss = 0
        
        # adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch_id-1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # training mode
        with tqdm(total=len(training_data_loader)) as t:
            for i_batch, batch in enumerate(training_data_loader):
                LR, HR_2_target, HR_4_target, HR_8_target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])

                LR = LR.to(params.device))
                HR_2_target = HR_2_target.to(params.device)
                HR_4_target = HR_4_target.to(params.device)
                HR_8_target = HR_8_target.to(params.device)

                optimizer.zero_grad()
                HR_2x, HR_4x, HR_8x = model(LR)
                
                loss_x2 = criterion(HR_2x, HR_2_target)
                loss_x4 = criterion(HR_4x, HR_4_target)
                loss_x8 = criterion(HR_8x, HR_8_target)

                loss = loss_x2 + loss_x4 + loss_x8
                running_loss += loss.data[0].item()

                loss.backward()
                optimizer.step()

                t.set_postfix(loss=running_loss/((i_batch+1)*params.batch_size))
                t.update()
            if epoch_id % params.valid_every == 0:
                utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_id, 0)
            
            

if __name__ == "__main__":
    main()

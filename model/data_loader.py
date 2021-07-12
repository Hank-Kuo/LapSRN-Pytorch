import os 
import random

from model.transform import LR_transform, HR_2_transform, HR_4_transform, HR_8_transform, Center_transform

from PIL import Image
import numpy as np
import torch.utils.data as data
import torchvision



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def augment_data(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    randint = np.random.randint(0, 4)
    if randint == 0:
        y = y.rotate(90)
    elif randint == 1:
        y = y.rotate(180)
    elif randint ==2:
        y = y.rotate(270)
    else:
        pass
    scale = random.uniform(0.5, 1)
    y = y.resize((int(y.size[0]*scale), int(y.size[1]*scale)), Image.BICUBIC)
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, crop_size):
        super(DatasetFromFolder, self).__init__()
        
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]

        self.LR_transform = LR_transform(crop_size)
        self.HR_2_transform = HR_2_transform(crop_size)
        self.HR_4_transform = HR_4_transform(crop_size)
        self.HR_8_transform = HR_8_transform(crop_size)


    def __getitem__(self, index):
        origin = augment_data(self.image_filenames[index])
        HR_8 = self.HR_8_transform(origin)
        HR_4 = self.HR_4_transform(HR_8)
        HR_2 = self.HR_2_transform(HR_8)
        LR = self.LR_transform(HR_8)

        to_tensor = torchvision.transforms.ToTensor()
        HR_8 = to_tensor(HR_8)
        return LR, HR_2, HR_4, HR_8

    def __len__(self):
        return len(self.image_filenames)


class TestDataset(data.Dataset):
    def __init__(self, image_dir, crop_size):
        super(TestDataset, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.path.listdir(image_dir) if is_image_file(x)]
        
        self.center_transform = Center_transform(crop_size)
        self.LR_transform = LR_transform(crop_size)
        self.HR_2_transform = HR_2_transform(crop_size)
        self.HR_4_transform = HR_4_transform(crop_size)
        self.HR_8_transform = HR_8_transform(crop_size)

    def __getitem__(self, index):        
        image = Image.open(self.image_filenames[index]).convert('YCbCr')
        image = self.Center_transform(image)
        y, _, _ = image.split()
        
        to_PIL = torchvision.transforms.ToPILImage()

        HR_8 = image
        HR_4 = to_PIL(self.HR_4_transform(image))
        HR_2 = to_PIL(self.HR_2_transform(image))

        LR = to_PIL(self.LR_transform(y))

        return LR, HR_2, HR_4, HR_8

    def __len__(self):
        return len(self.image_filenames)


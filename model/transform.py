from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, RandomCrop, RandomHorizontalFlip


def LR_transform(crop_size):
    return Compose([
        Scale(crop_size//8),
        ToTensor(),
    ])

def HR_2_transform(crop_size):
    return Compose([
        Scale(crop_size//4),
        ToTensor(),
    ])

def HR_4_transform(crop_size):
    return Compose([
        Scale(crop_size//2),
        ToTensor(),
    ])

def HR_8_transform(crop_size):
    return Compose([
        RandomCrop((crop_size, crop_size)),
        RandomHorizontalFlip(),
        # ToTensor(),
    ])


def Center_transform(crop_size):
    return Compose([
        CenterCrop(crop_size)
    ])


'''
class ImageTransforms(object):
    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        # Crop
        if self.split == 'train':
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # Sanity check
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img
'''
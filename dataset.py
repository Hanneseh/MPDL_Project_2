import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import float32, tensor
from torch.nn.functional import conv2d
from torchvision.transforms import functional as F

def collate_fn(examples):
    images, residuals, masks = zip(*examples)

    # Ensure images always have 3 channels
    images = [img if img.shape[0] == 3 else img.repeat((3, 1, 1)) for img in images]
    residuals = [res if res.shape[0] == 3 else res.repeat((3, 1, 1)) for res in residuals]
    
    # Add a singleton dimension to represent the channel dimension in masks
    masks = [mask.unsqueeze(0) for mask in masks]
    
    # Make sure images are always 512x512 by cropping or padding
    images = [F.resize(img, [512, 512]) if max(img.shape[1:]) > 512 else F.pad(img, (0, 512-img.shape[2], 0, 512-img.shape[1])) for img in images]
    masks = [F.resize(mask, [512, 512]) if max(mask.shape[1:]) > 512 else F.pad(mask, (0, 512-mask.shape[2], 0, 512-mask.shape[1])) for mask in masks]
    residuals = [F.resize(res, [512, 512]) if max(res.shape[1:]) > 512 else F.pad(res, (0, 512-res.shape[2], 0, 512-res.shape[1])) for res in residuals]

    # Stack images and masks into tensors
    images = torch.stack(images)
    masks = torch.stack(masks)
    residuals = torch.stack(residuals)
    return images, residuals, masks


class ImageDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path

        self.image_files = []
        self.mask_files = []

        for file_name in os.listdir(dir_path):
            if file_name.endswith('.realfake.webp'):
                self.image_files.append(os.path.join(dir_path, file_name))
            elif file_name.endswith('.mask.webp'):
                self.mask_files.append(os.path.join(dir_path, file_name))

        # Ensure that the masks and images are in the same order
        self.image_files.sort()
        self.mask_files.sort()

        filter1 = [[-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1]]
        filter2 =  [[0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0]]
        filter3 =  [[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0,-2, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / 12
        filter2 = np.asarray(filter2, dtype=float) / 4
        filter3 = np.asarray(filter3, dtype=float) / 2
        filters = np.asarray(( [[filter1, filter1, filter1], 
                                [filter2, filter2, filter2], 
                                [filter3, filter3, filter3]]))
        self.filters = tensor(filters, dtype=float32)


    def SRM(self, imgs):
        imgs = np.array(imgs, dtype=np.float32)
        imgs = np.einsum('klij->kjli', imgs)
        input = tensor(imgs, dtype=float32)
        op1 = conv2d(input, self.filters, stride=1, padding=2)

        op1 = op1[0]
        op1 = np.round(op1)
        op1[op1 > 2] = 2
        op1[op1 < -2] = -2
        return op1

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(image_path).convert('RGB')
        residual = self.SRM([np.asarray(image, dtype=np.float32)])
        mask = Image.open(mask_path).convert('RGB')

        # convert to numpy array
        mask = np.asarray(mask)
        # only take the first color channel (512, 512, 3) -> (512, 512)
        mask = mask[:,:,0]
        # convert mask to binary mask: Everything samaller than 255 is 0, everything else is 1
        mask = np.where(mask == 255, 1, 0)

        transform = transforms.ToTensor()
        image = transform(image)

        # Convert to tensor
        mask = torch.from_numpy(mask)
        # Change the data type of mask to Float
        mask = mask.float()

        return image, residual, mask
    

class FakeFakeDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path

        self.image_files = []
        self.mask_files = []

        for file_name in os.listdir(dir_path):
            if file_name.endswith('.fakefake.webp'):
                self.image_files.append(os.path.join(dir_path, file_name))
            elif file_name.endswith('.mask.webp'):
                self.mask_files.append(os.path.join(dir_path, file_name))

        # Ensure that the masks and images are in the same order
        self.image_files.sort()
        self.mask_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        # convert to numpy array
        mask = np.asarray(mask)

        # only take the first color channel (512, 512, 3) -> (512, 512)
        mask = mask[:,:,0]

        # convert mask to binary mask: Everything samaller than 255 is 0, everything else is 1
        mask = np.where(mask == 255, 1, 0)

        transform = transforms.ToTensor()
        image = transform(image)

        # Convert to tensor
        mask = torch.from_numpy(mask)

        # Change the data type of mask to Float
        mask = mask.float()

        return image, mask

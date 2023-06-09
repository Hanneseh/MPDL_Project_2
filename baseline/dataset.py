import os
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        transform = transforms.ToTensor()
        image = transform(image)
        mask = transform(mask)

        return image, mask

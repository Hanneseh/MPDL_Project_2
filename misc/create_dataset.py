from torch.utils.data import Dataset
from torchvision.transforms import Compose, PILToTensor
from PIL import Image
from torch import tensor, float32
from torch.nn.functional import conv2d
from os import listdir
from torchvision.transforms import ToTensor, Compose
import numpy as np
            

class data(Dataset):
    def __init__(self, datapath):
        """Creates a Datase

        Args:
            datapath (str): Path to the folder containing the data e.g.:
            /Users/pauladler/MPDL_Project_2_dev/data/train
        """
        self.datapath = datapath
        self.realfake, self.mask = self.returnpaths()
        self.length = len(self.mask)
        self.transform = Compose([
              ToTensor()
            ])

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


    def __getitem__(self, index):
        realfake = Image.open(self.realfake[index - 1]).convert("RGB")
        residual = self.SRM([np.asarray(realfake, dtype=np.float32)])
        realfake = self.transform(realfake)
        residual = realfake - residual
        mask = Image.open(self.mask[index - 1]).convert("L")
        mask = self.transform(mask)
        return realfake, residual, mask

    def __len__(self):
        return self.length
    
    def returnpaths(self):
        file_list = [self.datapath + "/" + str(file) for file in listdir(self.datapath)]
        realfake = [(file) for file in file_list if "realfake" in file]
        realfake.sort()
        mask = [(file) for file in file_list if "mask" in file]
        mask.sort()
        return realfake, mask
    
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



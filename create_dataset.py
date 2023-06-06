from torch.utils.data import Dataset
from torchvision.transforms import Compose, PILToTensor
from PIL import Image
from os import listdir
            

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
        self.transform = Compose([PILToTensor()])

    def __getitem__(self, index):
        realfake = Image.open(self.realfake[index - 1]).convert("RGB")
        realfake = self.transform(realfake)
        mask = Image.open(self.mask[index - 1]).convert("L")
        mask = self.transform(mask)
        return realfake, mask

    def __len__(self):
        return self.length
    
    def returnpaths(self):
        file_list = [self.datapath + "/" + str(file) for file in listdir(self.datapath)]
        realfake = [(file) for file in file_list if "realfake" in file]
        realfake.sort()
        mask = [(file) for file in file_list if "mask" in file]
        mask.sort()
        return realfake, mask



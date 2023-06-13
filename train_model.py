from torch.utils.data import DataLoader
from create_dataset import data
from training import train_model, val_epoch
from nix import NIX
import torch


if __name__== "__main__":
        
    # Set device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda_is_available():
        device = "cuda"

    device = torch.device(device)

    #Set Paths
    PATH_TRAIN = "/Users/pauladler/MPDL_Project_2/data/train"
    PATH_VAL = "/Users/pauladler/MPDL_Project_2/data/val"
    PATH_TEST = "/Users/pauladler/MPDL_Project_2/data/test"
    PATH_SAVE = "nix.pth"

    #Set Parameters for creating the Dataset
    num_workers = 5
    batch_size = 1

    # Set Parameters for model
    img_width, img_height = 512, 512

    # Set hyperparameters for training
    learning_rate = 0.0001

    train_data = data(PATH_TRAIN)
    val_data = data(PATH_VAL)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model = NIX(img_width, img_height)
    model = model.to(device)
    model = train_model(model, train_dataloader, val_dataloader, learning_rate, device)

    torch.save(model.state_dict(), PATH_SAVE)

    test_data = data(PATH_TEST)
    test_data = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    test_loss, test_iou = val_epoch(model, test_data, device)
    print("[INFO] Test loss: {:.6f}, Test IoU".format(test_loss, test_iou))

from torch.utils.data import DataLoader
from dataset import ImageDataset, collate_fn
from training import train_model, val_epoch
from model_definitions.nix_60 import NIX
import torch


if __name__== "__main__":
    device = "cuda:0"
    device = torch.device(device)

    #Set Paths
    PATH_TRAIN = "/home/guenthhx/MPDL2/data/train"
    PATH_VAL = "/home/guenthhx/MPDL2/data/val"
    PATH_TEST = "/home/guenthhx/MPDL2/data/test"
    PATH_SAVE = "nix_60_dataset2_inpainted.pth"

    #Set Parameters for creating the Dataset
    num_workers = 4
    batch_size = 8
    num_epochs = 100

    # Set Parameters for model
    img_width, img_height = 512, 512

    # Set hyperparameters for training
    learning_rate = 0.0001

    train_data = ImageDataset(PATH_TRAIN, "inpainted")
    val_data = ImageDataset(PATH_VAL, "inpainted")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    model = NIX(img_width, img_height)
    model = model.to(device)
    model = train_model(model, train_dataloader, val_dataloader, learning_rate, device, num_epochs)

    torch.save(model.state_dict(), PATH_SAVE)

    test_data = ImageDataset(PATH_TEST, "inpainted")
    test_data = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn, pin_memory=True)

    test_loss, test_iou = val_epoch(model, test_data, device)
    print("[INFO] Test loss: {:.6f}, Test IoU: {:.6f}".format(test_loss, test_iou))


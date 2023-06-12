import numpy as np
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch import float32, no_grad
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import jaccard_score


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.min_model = None

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.min_model = model
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def train_epoch(model, dataloader, opt, device):
    model.train()
    totaltrainloss = 0
    total_iou = 0
    for x, r, y in tqdm(dataloader):
        x, r, y = x.to(device, dtype=float32), r.to(device, dtype=float32), y.to(device, dtype=float32)

        opt.zero_grad()
        output = model(x, r)
        loss = sigmoid_focal_loss(output, y, reduction="mean")

        loss.backward()
        opt.step()

        totaltrainloss += loss.item()

        pred = (output > 0.5).int()
        y = (y > 0.5).int()
        total_iou += jaccard_score(y.flatten().cpu().numpy(), pred.flatten().cpu().numpy())

    totaltrainloss = totaltrainloss/len(dataloader)
    total_iou = total_iou / len(dataloader)
    return model, totaltrainloss, total_iou


def val_epoch(model, dataloader, device):
    model.eval()
    totalvalloss = 0
    total_iou = 0
    with no_grad():
        for x, r, y in tqdm(dataloader):
            x, r, y = x.to(device, dtype=float32), r.to(device, dtype=float32), y.to(device, dtype=float32)
            output = model(x, r)
            totalvalloss += sigmoid_focal_loss(output, y, reduction="mean").item()
            pred = (output > 0.5).int()
            y = (y > 0.5).int()
            total_iou += jaccard_score(y.flatten().cpu().numpy(), pred.flatten().cpu().numpy())
    totalvalloss = totalvalloss/len(dataloader)
    total_iou = total_iou / len(dataloader)
    return totalvalloss, total_iou


def train_model(model, train_data, val_data ,lr, device, weight_decay=0.0005, patience=3):
    """Full Training function for the nix

    Args:
        model (torch.nn.Module): nix model
        train_data (torch.utils.data.DataLoader): DataLoader containing the train data
        val_data (torch.utils.data.DataLoader): DataLoader containing the val data
        lr (int): learning rate for training
        device (torch.device): Device for training
        weight_decay (float, optional): l2 regularization. Defaults to 0.0001.
        patience (int, optional): patience for the early_stopper. Defaults to 3

    Returns:
        torch.nn.Module: A trained model
    """
    print("[INFO] Training with lr: {}".format(lr))
    optim = Adam(model.parameters(), lr, weight_decay=weight_decay)
    epoch = 0
    early_stopper = EarlyStopper(patience=patience)
    while True:
        epoch += 1
        print("[INFO] Epoch: {}".format(epoch))
        model, train_loss, train_iou = train_epoch(model, train_data, optim, device)
        print("Train loss: {:.6f}, Train IoU: {:.6f}".format(train_loss, train_iou))
        val_loss, val_iou = val_epoch(model, val_data, device)
        print("Val loss: {:.6f}, Val IoU: {:.6f}".format(val_loss, val_iou))
        if early_stopper.early_stop(val_loss, model):
            print("[INFO] End training with lr {} at epoch {}".format(lr, epoch))
            break
    return early_stopper.min_model

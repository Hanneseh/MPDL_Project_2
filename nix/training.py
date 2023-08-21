import numpy as np
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch import float32, no_grad, sigmoid
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import torch


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


        pred = (sigmoid(output) > 0.5).int()
        y = (sigmoid(y) > 0.5).int()

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

            pred = (sigmoid(output) > 0.5).int()
            y = (sigmoid(y) > 0.5).int()

            total_iou += jaccard_score(y.flatten().cpu().numpy(), pred.flatten().cpu().numpy())

    totalvalloss = totalvalloss/len(dataloader)
    total_iou = total_iou / len(dataloader)
    return totalvalloss, total_iou


def train_model(model, train_data, val_data ,lr, device, num_epochs, weight_decay=0.001, patience=5, save_intervall=10):
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
    train_losses = []
    val_losses = []
    train_iou_list = []
    val_iou_list = []
    
    while True:
        epoch += 1
        print("[INFO] Epoch: {}".format(epoch))
        model, train_loss, train_iou = train_epoch(model, train_data, optim, device)
        print("Train loss: {:.6f}, Train IoU: {:.6f}".format(train_loss, train_iou))

        val_loss, val_iou = val_epoch(model, val_data, device)
        print("Val loss: {:.6f}, Val IoU: {:.6f}".format(val_loss, val_iou))

        #save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_iou_list.append(train_iou)
        val_iou_list.append(val_iou)

        #Secure-Save in specific intervall
        if epoch % save_intervall == 0: 
            save_path = "nix_epoch_{}".format(epoch)
            save_name = save_path+".pth"
            torch.save(model.state_dict(), save_name)

        if early_stopper.early_stop(val_loss, model):
            print("[INFO] early_stop: End training with lr {} at epoch {}".format(lr, epoch))
            model = early_stopper.min_model
            break

        if num_epochs == epoch: 
            print("[INFO] epoch_stop: End training with lr {} at epoch {}".format(lr, epoch))
            model = model
            break
    
    #log losses
    with open("loss.txt", "w") as file:
         for epoch, train_loss, val_loss, train_iou, val_iou in zip(range(1, epoch+1), train_losses, val_losses, train_iou_list, val_iou_list):
             file.write("Epoch {}: Train loss: {}, Val loss: {}, Train IoU: {}, Val Iou: {} \n".format(epoch, train_loss, val_loss, train_iou, val_iou))

    #Plot Loss-Curve
    epochs = range(1, epoch+1)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.show()


    return model

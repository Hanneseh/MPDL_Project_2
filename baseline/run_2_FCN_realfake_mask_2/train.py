from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from earlystopper import EarlyStopper
from dataset import ImageDataset
import os
from torchvision.transforms import functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss
from  torch.utils.data import Subset

def collate_fn(examples):
    images, masks = zip(*examples)

    # Ensure images always have 3 channels
    images = [img if img.shape[0] == 3 else img.repeat((3, 1, 1)) for img in images]
    
    # Add a singleton dimension to represent the channel dimension in masks
    masks = [mask.unsqueeze(0) for mask in masks]
    
    # Make sure images are always 512x512 by cropping or padding
    images = [F.resize(img, [512, 512]) if max(img.shape[1:]) > 512 else F.pad(img, (0, 512-img.shape[2], 0, 512-img.shape[1])) for img in images]
    masks = [F.resize(mask, [512, 512]) if max(mask.shape[1:]) > 512 else F.pad(mask, (0, 512-mask.shape[2], 0, 512-mask.shape[1])) for mask in masks]

    # Stack images and masks into tensors
    images = torch.stack(images)
    masks = torch.stack(masks)

    return images, masks

train_losses = []
train_iou = []

eval_losses = []
eval_iou = []

def train(epoch, data):
    print('\nEpoch : %d' % epoch)
    model.train()
    running_loss = 0
    total_iou = 0
    for batch in tqdm(data):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)["out"]
        loss = criterion(outputs, labels, reduction='mean')

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate IoU
        pred = (torch.sigmoid(outputs) > 0.5).int()
        labels = (torch.sigmoid(labels) > 0.5).int()
        total_iou += jaccard_score(labels.flatten().cpu().numpy(), pred.flatten().cpu().numpy())

    train_loss = running_loss / len(data)
    iou = total_iou / len(data)

    train_iou.append(iou)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | IoU: %.3f' % (train_loss, iou))

def val(data):
    model.eval()
    running_loss = 0
    total_iou = 0
    with torch.no_grad():
        for batch in data:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)["out"]
            loss = criterion(outputs, labels, reduction='mean')

            running_loss += loss.item()

            # Calculate IoU
            pred = (torch.sigmoid(outputs) > 0.5).int()
            labels = (torch.sigmoid(labels) > 0.5).int()
            total_iou += jaccard_score(labels.flatten().cpu().numpy(), pred.flatten().cpu().numpy())

    val_loss = running_loss / len(data)
    iou = total_iou / len(data)

    eval_iou.append(iou)
    eval_losses.append(val_loss)
    print('Val Loss: %.3f | IoU: %.3f' % (val_loss, iou))
    return val_loss

def plot_and_save(train_iou, eval_iou, train_loss, eval_loss, parent_folder, epoch_nr):
    
    plt.figure(figsize=(12, 6))

    # Plot training and validation IoU scores
    plt.subplot(1, 2, 1)
    plt.plot(train_iou, '-o')
    plt.plot(eval_iou, '-o')
    plt.xlabel('epoch')
    plt.ylabel('IoU score')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid IoU Scores')

    # Plot training and validation losses
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, '-o')
    plt.plot(eval_loss, '-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')

    plt.tight_layout()

    # Create parent directory if it does not exist
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    # Save the plot
    plt.savefig(f"{parent_folder}/plot_epoch_{epoch_nr}.png")

# Parameters
data_dir = "data/laion_subset_10k_splits_train"
checkpoint_dir = "baseline/checkpoints"
docs_path = "baseline/docs"
num_epochs = 100
batch_size = 8
learning_rate = 0.0001
num_workers = 2
patience = 5

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
# Modify the output layer
model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
# path_to_pretrained_weights = "baseline/checkpoints/baseline_epoch_31.pth"
# model.load_state_dict(torch.load(path_to_pretrained_weights))
model = model.to(device)

# datasets
train_dataset = ImageDataset(os.path.join(data_dir, 'train'))
test_dataset = ImageDataset(os.path.join(data_dir, 'test'))
val_dataset = ImageDataset(os.path.join(data_dir, 'val'))

# # for dev purposes: reduce the number of images in the datasets to only 10
# train_dataset = Subset(train_dataset, range(30))
# test_dataset = Subset(test_dataset, range(10))
# val_dataset = Subset(val_dataset, range(10))

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

# Define the loss function and optimizer
criterion = sigmoid_focal_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
stopper = EarlyStopper(patience=patience, min_delta=0)

# Train the model
for epoch in range(num_epochs):
    train(epoch, train_loader)
    val_loss = val(val_loader)
    PATH = checkpoint_dir + '/baseline_epoch_{}.pth'.format(epoch)
    torch.save(model.state_dict(), PATH)
    plot_and_save(train_iou, eval_iou, train_losses, eval_losses, docs_path, epoch)
    if stopper.early_stop(val_loss, model):
        model = stopper.min_model
        print("early stop")
        break

# safe model state
PATH = checkpoint_dir + '/baseline_final.pth'
torch.save(model.state_dict(), PATH)
plot_and_save(train_iou, eval_iou, train_losses, eval_losses, docs_path, epoch)

# test final model on test set
print("Test final model on test set")
test_loss = val(test_loader)
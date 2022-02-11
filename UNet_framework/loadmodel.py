from model import UNet, WeightMap
from dataloader import Cell_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os

image_size = 572
# root directory of project
root_dir = os.getcwd()
# training batch size
batch_size = 30

data_dir = os.path.join(root_dir, "data/cells")

trainset = Cell_data(data_dir=data_dir, size=image_size)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

testset = Cell_data(data_dir=data_dir, size=image_size, train=False)
testloader = DataLoader(testset, batch_size=4)

model = UNet()
model.load_state_dict(torch.load("checkpoint.pt"))
model.to("cuda:0")
model.eval()

output_masks = []
output_labels = []

with torch.no_grad():
    for i in range(testset.__len__()):
        image, labels = testset.__getitem__(i)
        labels = labels.squeeze()

        input_image = image.unsqueeze(0).to("cuda:0")
        pred = model(input_image)

        output_mask = torch.argmax(pred, dim=1).cpu().squeeze(0).numpy()

        crop = (labels.shape[0] - output_mask.shape[0]) // 2
        labels = labels[crop:-crop, crop:-crop].numpy()

        output_masks.append(output_mask)
        output_labels.append(labels)

fig, axes = plt.subplots(testset.__len__(), 2)

for i in range(testset.__len__()):
    axes[i, 0].imshow(output_labels[i])
    axes[i, 0].axis("off")
    axes[i, 1].imshow(output_masks[i])
    axes[i, 1].axis("off")
plt.show()

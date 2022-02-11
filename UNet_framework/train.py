from model import UNet, WeightMap
from dataloader import Cell_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# import any other libraries you need below this line
import math

# Paramteres

# learning rate
lr = 1e-3
# number of training epochs
epoch_n = 40
# input image-mask size
image_size = 572
# root directory of project
root_dir = os.getcwd()
# training batch size
batch_size = 4
# use checkpoint model for training
load = False
# use GPU for training
gpu = True

use_weightmap = False

data_dir = os.path.join(root_dir, "data/cells")

trainset = Cell_data(data_dir=data_dir, size=image_size)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

testset = Cell_data(data_dir=data_dir, size=image_size, train=False)
testloader = DataLoader(testset, batch_size=4)

device = torch.device("cuda:0" if gpu else "cpu")

model = UNet().to("cuda:0").to(device)
get_weight = WeightMap().to("cuda:0").to(device)

if load:
    print("loading model")
    model.load_state_dict(torch.load("checkpoint.pt"))

criterion = nn.CrossEntropyLoss()
criterion_train = nn.CrossEntropyLoss(reduction="none")

optimizer = optim.Adam(
    model.parameters(), lr=lr, weight_decay=0
)  # 1e-4 /1e-5 / no weightdecay

model.train()

loss_history = []
accuracy_history = []
train_loss_history = []

train_loss_min = math.inf
test_accu_max = 0

for e in range(epoch_n):
    print("--------\nepoch %d\n--------" % (e + 1))
    epoch_loss = 0
    model.train()
    for i, data in enumerate(trainloader):
        image, label = data

        image = image.to(device)
        label = label.long().to(device)

        pred = model(image)

        crop = int((label.shape[-1] - pred.shape[-1]) / 2)
        label = label[:, :, crop:-crop, crop:-crop]

        if use_weightmap:
            weight = get_weight(label)
            loss = criterion_train(pred, label.squeeze())
            loss = weight * loss
            loss = torch.sum(loss.flatten(start_dim=1), axis=0)
            loss = torch.mean(loss)
        else:
            loss = criterion(pred, label.squeeze())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        # print("    batch %d --- Loss: %.4f" % (i, loss.item() / batch_size))
    print(
        "Epoch %d / %d --- Loss: %.4f"
        % (e + 1, epoch_n, epoch_loss / trainset.__len__())
    )

    train_loss_history.append(epoch_loss / len(trainset))

    if epoch_loss < train_loss_min:
        train_loss_min = epoch_loss
        torch.save(model.state_dict(), "checkpoint.pt")

    model.eval()

    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            image, label = data

            image = image.to(device)
            label = label.long().to(device)

            pred = model(image)

            crop = int((label.shape[-1] - pred.shape[-1]) / 2)
            label = label[:, :, crop:-crop, crop:-crop]

            loss = criterion(pred, label.squeeze())

            total_loss += loss.item()
            pred_labels = torch.argmax(pred, dim=1)

            sub_total = label.shape[0] * label.shape[3] * label.shape[2]
            sub_correct = (pred_labels == label.squeeze()).sum().item()
            total += sub_total
            correct += sub_correct

            # print(
            #     "    test batch %d loss: %.4f, accuracy: %4f"
            #     % (i, loss.item() / batch_size, sub_correct / sub_total)
            # )

        print(
            "Accuracy: %.4f ---- Loss: %.4f"
            % (correct / total, total_loss / testset.__len__())
        )

        accuracy_history.append(correct / total)
        loss_history.append(total_loss / testset.__len__())

        if correct / total > test_accu_max:
            test_accu_max = correct / total
            torch.save(model.state_dict(), "best.pt")


# testing and visualization

model = UNet()
model.load_state_dict(torch.load("checkpoint.pt"))
model.to(device)
model.eval()

output_masks = []
output_labels = []

with torch.no_grad():
    for i in range(testset.__len__()):
        image, labels = testset.__getitem__(i)
        labels = labels.squeeze()

        input_image = image.unsqueeze(0).to(device)
        pred = model(input_image)

        output_mask = torch.argmax(pred, dim=1).cpu().squeeze(0).numpy()

        crop = (labels.shape[0] - output_mask.shape[0]) // 2
        labels = labels[crop:-crop, crop:-crop].numpy()

        output_masks.append(output_mask)
        output_labels.append(labels)

fig, axes = plt.subplots(testset.__len__(), 2, figsize=(40, 40))

for i in range(testset.__len__()):
    axes[i, 0].imshow(output_labels[i])
    axes[i, 0].axis("off")
    axes[i, 1].imshow(output_masks[i])
    axes[i, 1].axis("off")
plt.show()
plt.figure()
plt.plot(loss_history, label="test loss")
plt.plot(accuracy_history, label="test accuracy")
plt.plot(train_loss_history, label="train loss")
plt.legend()
plt.show()

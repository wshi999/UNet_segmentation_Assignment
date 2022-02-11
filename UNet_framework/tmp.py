from dataloader import Cell_data
import os
from PIL import Image, ImageOps
import torch
from torchvision import transforms

root_dir = os.getcwd()
data_dir = os.path.join(root_dir, "data/cells")
image_size = 1024

testset = Cell_data(data_dir=data_dir, size=image_size, train=False)
trainset = Cell_data(data_dir=data_dir, size=image_size, train=True)

for i in range(len(trainset)):
    img, label = trainset[i]
    print(torch.count_nonzero(label) / (label.shape[1] * label.shape[2]))


print("")
# path = "C:\\Users\\sang_sang\\Desktop\\733\\UNet_segmentation_Assignment\\data\\cells\\labels\\BMMC_48.bmp"
# p = Image.open(path)
# pp = t(p)
# rp = r(pp)
# rp[rp != 0] = 1
# rp = rp.long()


# for j in range(140, 600):
#     x = j
#     x = x - 4
#     x /= 2
#     if x % 1 != 0:
#         continue
#     x = x - 4
#     x /= 2
#     if x % 1 != 0:
#         continue
#     x = x - 4
#     x /= 2
#     if x % 1 != 0:
#         continue
#     x = x - 4
#     x /= 2
#     if x % 1 != 0:
#         continue
#     print(j)

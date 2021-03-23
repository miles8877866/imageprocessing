# -*- coding: utf-8 -*-
import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.image as mpimg
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms as transforms
img = mpimg.imread("C:/Users/as722/Desktop/tests/day0716/pca_poker_data/testing/0/1.bmp")

class MNISTDataset(Dataset): 
    def __init__(self,root,train,transform): 
        self.transform = transform
        self.image_files = []
        if train:
            dic = 'training'
        else:
            dic = 'testing'
        for label in os.listdir(root+dic): 
            for r, _, f in os.walk(root+dic+'/'+label):
                for item in range(len(f)):
                    self.image_files.append((r+'/'+f[item],int(label)))

    def __getitem__(self, index):    
        img_name, label = self.image_files[index]
        img = mpimg.imread(img_name)
        img = self.transform(img)
        return[img,label]

    def __len__(self): 
        return len(self.image_files)

transform = transforms.Compose([transforms.ToTensor()])
img,label = MNISTDataset(r'C:/Users/as722/Desktop/tests/day0716/pca_poker_data/',True,transform).__getitem__(51)
# plt.figure()
# plt.imshow(img)
trainset = MNISTDataset(r'C:/Users/as722/Desktop/tests/day0716/pca_poker_data/',train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

testset = MNISTDataset(r'C:/Users/as722/Desktop/tests/day0716/pca_poker_data/',train=False,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=True)

classes = ['0', '1', '2', '3']
def show_batch(data):
    imgs,labels = data
    grid = utils.make_grid(imgs,nrow=5)
#     print(grid.numpy().shape)
#     print(grid.numpy().transpose((1, 2, 0)).shape)
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


classes = ('0', '1', '2', '3')

for i, data in enumerate(testloader):
    if(i<4):
        show_batch(data)
    else:
        break
###Net

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ###encoder_layers
        self.fc1 = nn.Conv2d(1, 10, 3, 1, 1)
        self.fc2 = nn.Conv2d(10, 15, 3, 1, 1)
        self.fc3 = nn.Conv2d(15, 30, 3, 1, 1)
        ###decoder_layer
        self.ct1 = nn.ConvTranspose2d(30, 15, 3, 1, 1)
        self.ct2 = nn.ConvTranspose2d(15, 10, 3, 1, 1)
        self.ct3 = nn.ConvTranspose2d(10, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.size()
        # x = x.view(b,c*h*w)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.ct1(x)
        x = self.ct2(x)
        x = self.ct3(x)
        x = self.sigmoid(x)
        return x

net = Net()
# print(net)
# for p in enumerate(net.parameters()):
#     print(p[1].size())
    
# specify loss function
criterion = nn.MSELoss()
# specify loss function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

###training
# =============================================================================
n_epochs = 100
for epoch in range(n_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
      ###################
      # train the model #
      ###################
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 4 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.10f' % (epoch+1 , i , running_loss / 4))
            running_loss = 0.0

print('Finished Training')
# =============================================================================
# obtain one batch of test images
dataiter = iter(testloader)
images, labels = dataiter.next()
# get sample outputs
output = net(images)
# prep images for display
# images = images.numpy()

# # output is resized into a batch of iages
# output = output.view(4, 1, 96, 71)
# # use detach when it's an output that requires_grad
# output = output.detach().numpy()

for i in range(4):
    show = output[i, 0, :, :]
    origin = images[i, 0, :, :]
    plt.figure()
    plt.imshow(origin.detach().numpy(),cmap='gray')
    plt.figure()
    plt.imshow(show.detach().numpy(),cmap='gray')
plt.show()


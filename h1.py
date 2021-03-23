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

def show_batch(data):
    imgs,labels = data
    grid = utils.make_grid(imgs,nrow=5)
#     print(grid.numpy().shape)
#     print(grid.numpy().transpose((1, 2, 0)).shape)

# plt.figure()
# plt.imshow(grid.numpy().transpose((1, 2, 0)))
# plt.title('')

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
        self.fc1 = nn.Linear(6816, 1000)
        self.fc2 = nn.Linear(1000, 512)
        self.fc3 = nn.Linear(512, 4)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.size()
        x = x.view(b,c*h*w)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

net = Net()
for p in enumerate(net.parameters()):
    print(p[1].size())
###Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

###training
for epoch in range(200):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        
        outputs = net(inputs)
        print(labels)
        print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.10f' % (epoch+1 , i , running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

###save weight
w_path = r'C:/Users/as722/Desktop/tests/day0716/pca_poker_data/w_poker_net.pth'
#m_path = './m_mnist_net.pth'
#torch.save(net.state_dict(), w_path)
#torch.save(net, m_path)

###load weight
#net = Net()
#net.load_state_dict(torch.load(w_path))

###testing
dataiter = iter(testloader)
data = dataiter.next()
images, labels = data
show_batch(data)

outputs = net(images)
outputs.size()


_, predicted = torch.max(outputs, 1)
print(_)
print(predicted)

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 52
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)      
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
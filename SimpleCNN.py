import numpy as np
import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import time
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name())

data = np.load('/blue/eee4773/eric.fowler/Final-Project/Data/data_train.npy').T
labels = np.loadtxt('/blue/eee4773/eric.fowler/Final-Project/Data/correct_labels.npy',delimiter=',')

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

nimages = len(train_labels)

class data_and_labels(Dataset):
    def __init__(self,data,labels,transform = None):
        self.transform = transform
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(labels)
    def __getitem__(self,index):
        image = data[index,:]
        label = torch.tensor(int(labels[index]))

        if self.transform:
            image = self.transform(image)
        return (image,label)

dataset = data_and_labels(train_data,train_labels,transform = transforms.ToTensor())
print(np.shape(dataset))
print(np.shape(train_data))
print(nimages)
print(int(nimages*0.8))
print(nimages-int(nimages*0.8))
train_set,valid_set = torch.utils.data.random_split(dataset,[int(nimages*0.8),nimages-int(nimages*0.8)])

# #train_data_tensor = torch.from_numpy(train_data).float().to(device)
# #valid_data_tensor = 
# test_data_tensor = torch.from_numpy(test_data).float().to(device)
# #train_labels_tensor = torch.from_numpy(train_labels).float().to(device)
# test_labels_tensor = torch.from_numpy(test_labels).float().to(device)

train_loader = DataLoader(dataset=train_set,batch_size=20,shuffle=True)
test_loader = DataLoader(dataset=valid_set,batch_size=20,shuffle=True)


lr = 0.001
momentum = 0.9
num_epochs = 20

#conv1
c1inputsize = 1 #greyscale
c1outputsize = 3 #idk
c1kernelsize = 5 #idk
#pool1
p1ks = 5 #kernel size
p1s = p1ks #stride size
#conv2
c2is = c1outputsize
c2os = 10
c2ks = 5
#fully connected 1
fc1is = c2os*c1kernelsize*c2ks
fc1os = 80
#fully connected 2
fc2is = fc1os
fc2os = 40
#fully connected 3
fc3is = fc2os
fc3os = 22

# class Model(nn.Module):
#     def __init__(self,layers):
#         super().__init__()

#         self.activation = nn.Tanh()

#         self.loss_function = nn.MSELoss(reduction='mean')

#         self.conv1 = nn.Conv2d(c1inputsize,c1outputsize,c1kernelsize)
#         self.pool = nn.MaxPool2d(p1ks,p1s)
#         self.conv2 = nn.Conv2d(c2is,c2os,c2ks)
#         self.fc1 = nn.Linear(fc1is,fc1os)
#         self.fc2 = nn.Linear(fc2is,fc2os)
#         self.fc3 = nn.Linear(fc3is,fc3os)
#     def forward(self,x):
#         x = self.pool(F.relu)
model = nn.Sequential(
    nn.Conv2d(c1inputsize,c1outputsize,c1kernelsize),
    nn.MaxPool2d(p1ks,p1s),
    nn.ReLU(),
    nn.Conv2d(c2is,c2os,c2ks),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(fc1is,fc1os),
    nn.Linear(fc2is,fc2os),
    nn.Linear(fc3is,fc3os)
)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum)

train_losses = []
valid_losses = []

for epoch in range(1,num_epochs+1):
    train_loss=0.0
    valid_loss=0.0
    model.train()
    for image, label in train_loader:
        image = image.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        prediction=  model(image)
        loss = loss_fn(prediction,label)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*image.size(0)
    model.eval()
    for image,label in test_loader:
        image = image.cuda()
        label = label.cuda()
        prediction = model(image)
        loss = loss_fn(prediction,label)
        valid_loss+=loss.item()*image.size(0)
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(test_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print('Epoch:{} Train Loss:{:.4f} valid Loss:{:.4f}'.format(epoch,train_loss,valid_loss))
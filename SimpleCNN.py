import numpy as np
import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import time

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

class Model(nn.Module):
    def __init__(self,layers):
        super().__init__()

        self.activation = nn.Tanh()

        self.loss_function = nn.MSELoss(reduction='mean')

        self.conv1 = nn.Conv2d(c1inputsize,c1outputsize,c1kernelsize)
        self.pool = nn.MaxPool2d(p1ks,p1s)
        self.conv2 = nn.Conv2d(c2is,c2os,c2ks)
        self.fc1 = nn.Linear(fc1is,fc1os)
        self.fc2 = nn.Linear(fc2is,fc2os)
        self.fc3 = nn.Linear(fc3is,fc3os)

cnn_model = Sequential([
    Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),
    MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
    Dropout(0.2),
    Flatten(), # flatten out the layers
    Dense(32,activation='relu'),
    Dense(10,activation = 'softmax')
    
])
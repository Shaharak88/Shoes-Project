import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 318178704

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = 8
        self.n = n
        kernel_size = 5
        padding = int((kernel_size - 1) / 2)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=padding)
        self.conv3 = nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=padding)
        self.conv4 = nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=padding)
        self.fc1 = nn.Linear(8*n*28*14, 100)
        self.fc2 = nn.Linear(100, 2)
        # TODO: complete this method
    
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        #print("check3")
        out = self.conv1(inp)
        out = F.relu(out)
        out = self.pool(out)
        
        
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool(out)
        
        out = self.conv3(out)
        out = F.relu(out)
        out = self.pool(out)
        
        
        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool(out)
        
        out_size = out.size(1)*out.size(2)*out.size(3)
        #print(out.shape)
        out = out.reshape(-1, self.n*28*14*8)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        #print(out.shape)
        # TODO: complete this function
        return out #is the shape of out.shape should be (N,2)?

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        n = 8
        self.n = n
        self.kernel_size = 5
        kernel_size = self.kernel_size
        padding = int((kernel_size - 1) / 2)
        self.conv1 = nn.Conv2d(in_channels=6,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=padding)
        self.conv3 = nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=padding)
        self.conv4 = nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=padding)
        self.fc1 = nn.Linear(8*n*14*14, 100)
        self.fc2 = nn.Linear(100, 2)
        # TODO: complete this method

    # TODO: complete this class
    def forward(self,inp):# Do NOT change the signature of this functio
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # TODO start by changing the shape of the input to (N,6,224,224)
        # TODO: complete this function
        # Split the images along the height dimension
        #print(inp.shape)
        upper_half, lower_half = inp.split(224, dim=2)

        # Concatenate the images along the channel dimension
        combined = torch.cat([upper_half, lower_half], dim=1)
        #print(f"Output tensor shape: {combined.shape}")
        

        
        

        # Pass the combined images through the convolutional layers
        x = self.pool(F.relu(self.conv1(combined)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the tensor for the fully connected layers
        x = x.reshape(-1, self.n*14*14*8)

        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(x.shape)

        return x
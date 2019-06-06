import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
import pdb


class Network_view_1(nn.Module):
    def __init__(self):
        super(Network_view_1, self).__init__()

        self.conv1 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )        
        self.conv2 = nn.Sequential(
             nn.Conv2d(16,64,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.conv3 = nn.Sequential(
             nn.Conv2d(64,128,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.conv4 = nn.Sequential(
             nn.Conv2d(128,256,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.fc = nn.Sequential(
             nn.Linear(256,512),
             nn.Linear(512,10)
            )
    def forward(self, input1):
        X1 = self.conv1(input1)
#        pdb.set_trace()
        X2 = self.conv2(X1)
        X3 = self.conv3(X2)
        X4 = self.conv4(X3)
        X4 = X4.reshape(X4.size(0), -1)
        output = self.fc(X4)
        return output


class Network_view_2(nn.Module):
    def __init__(self):
        super(Network_view_2, self).__init__()

        self.conv1_1 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1_2 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )        
        self.conv2 = nn.Sequential(
             nn.Conv2d(32,64,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.conv3 = nn.Sequential(
             nn.Conv2d(64,128,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.conv4 = nn.Sequential(
             nn.Conv2d(128,256,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.fc = nn.Sequential(
             nn.Linear(256,512),
             nn.Linear(512,10)
            )
    def forward(self, input1, input2):
        X1_1 = self.conv1_1(input1)
        X1_2 = self.conv1_2(input2)
        X1 = torch.cat((X1_1, X1_2), 1)
#        pdb.set_trace()
        X2 = self.conv2(X1)
        X3 = self.conv3(X2)
        X4 = self.conv4(X3)
        X4 = X4.reshape(X4.size(0), -1)
        output = self.fc(X4)
        return output

class Network_view_3(nn.Module):
    def __init__(self):
        super(Network_view_3, self).__init__()

        self.conv1_1 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1_2 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1_3 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )        
        self.conv2_1 = nn.Sequential(
             nn.Conv2d(32,64,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2_2 = nn.Sequential(
             nn.Conv2d(16,32,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.conv3 = nn.Sequential(
             nn.Conv2d(96,128,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.conv4 = nn.Sequential(
             nn.Conv2d(128,256,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.fc = nn.Sequential(
             nn.Linear(256,512),
             nn.Linear(512,10)
            )
    def forward(self, input1, input2, input3):
        X1_1 = self.conv1_1(input1)
        X1_2 = self.conv1_2(input2)
        X1_3 = self.conv1_3(input3)
        X1 = torch.cat((X1_1, X1_2), 1)
#        X1 = X1_1 + X1_2
#        pdb.set_trace()
        X2_1 = self.conv2_1(X1)
        X2_2 = self.conv2_2(X1_3)
#        pdb.set_trace()        
        X2 = torch.cat((X2_1, X2_2), 1) 
        X3 = self.conv3(X2)
        X4 = self.conv4(X3)
        X4 = X4.reshape(X4.size(0), -1)
        output = self.fc(X4)
        return output
        
class Network_view_4(nn.Module):
    def __init__(self):
        super(Network_view_4, self).__init__()

        self.conv1_1 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1_2 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1_3 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1_4 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )        
        self.conv2_1 = nn.Sequential(
             nn.Conv2d(32,64,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2_2 = nn.Sequential(
             nn.Conv2d(16,32,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        ) 
        self.conv2_3 = nn.Sequential(
             nn.Conv2d(16,32,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )               
        self.conv3_1 = nn.Sequential(
             nn.Conv2d(96,128,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv3_2 = nn.Sequential(
             nn.Conv2d(32,64,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.conv4 = nn.Sequential(
             nn.Conv2d(192,256,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.fc = nn.Sequential(
             nn.Linear(256,512),
             nn.Linear(512,10)
            )
    def forward(self, input1, input2, input3, input4):
        X1_1 = self.conv1_1(input1)
        X1_2 = self.conv1_2(input2)
        X1_3 = self.conv1_3(input3)
        X1_4 = self.conv1_4(input4)
        
        X1 = torch.cat((X1_1, X1_2), 1)
        
        X2_1 = self.conv2_1(X1)
        X2_2 = self.conv2_2(X1_3)
        X2_3 = self.conv2_3(X1_4)
#        pdb.set_trace()        
        X2 = torch.cat((X2_1, X2_2), 1) 
        
        X3_1 = self.conv3_1(X2)
        X3_2 = self.conv3_1(X2_3)
        X3 = torch.cat((X3_1, X3_2), 1)
        X4 = self.conv4(X3)
        X4 = X4.reshape(X4.size(0), -1)
        output = self.fc(X4)
        return output
        
class Network_view_5(nn.Module):
    def __init__(self):
        super(Network_view_5, self).__init__()

        self.conv1_1 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1_2 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1_3 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1_4 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1_5 = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )        
        self.conv2_1 = nn.Sequential(
             nn.Conv2d(32,64,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2_2 = nn.Sequential(
             nn.Conv2d(16,32,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        ) 
        self.conv2_3 = nn.Sequential(
             nn.Conv2d(16,32,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2_4 = nn.Sequential(
             nn.Conv2d(16,32,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )               
        self.conv3_1 = nn.Sequential(
             nn.Conv2d(96,128,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv3_2 = nn.Sequential(
             nn.Conv2d(32,64,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv3_3 = nn.Sequential(
             nn.Conv2d(32,64,kernel_size=6,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.conv4_1 = nn.Sequential(
             nn.Conv2d(192,256,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv4_2 = nn.Sequential(
             nn.Conv2d(64,128,kernel_size=5,stride=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2,stride=2)
        )                
        self.fc = nn.Sequential(
             nn.Linear(384,512),
             nn.Linear(512,10)
            )
    def forward(self, input1, input2, input3, input4, input5):
        
#        pdb.set_trace()
        X1_1 = self.conv1_1(input1)
        X1_2 = self.conv1_2(input2)
        X1_3 = self.conv1_3(input3)
        X1_4 = self.conv1_4(input4)
        X1_5 = self.conv1_5(input5)
        
        X1 = torch.cat((X1_1, X1_2), 1)
        
        X2_1 = self.conv2_1(X1)
        X2_2 = self.conv2_2(X1_3)
        X2_3 = self.conv2_3(X1_4)
        X2_4 = self.conv2_4(X1_5)
#        pdb.set_trace()        
        X2 = torch.cat((X2_1, X2_2), 1) 
        
        X3_1 = self.conv3_1(X2)
        X3_2 = self.conv3_2(X2_3)
        X3_3 = self.conv3_3(X2_4)
        
        X3 = torch.cat((X3_1, X3_2), 1)
        X4_1 = self.conv4_1(X3)
        X4_2 = self.conv4_2(X3_3)
        X4 = torch.cat((X4_1, X4_2), 1)
        
        X4 = X4.reshape(X4.size(0), -1)
        output = self.fc(X4)
        return output

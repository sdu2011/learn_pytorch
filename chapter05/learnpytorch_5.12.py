import torch
import torch.nn as nn
import sys
sys.path.append("..") 
import learntorch_utils
import time
import torchvision.models as models

net = models.DenseNet(growth_rate=32,block_config=(6,12,24,16),num_init_features=64,bn_size=4)
X = torch.randn((1,3,224,224))
features = net.features
features(X)
# for name,module in features.named_children():
#     print(name,module)
#     break

class DenseLayer(nn.Module):
    def __init__(self,in_channels,bottleneck_size,growth_rate):
        super(DenseLayer,self).__init__()
        count_of_1x1 = bottleneck_size*growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_channels,count_of_1x1,kernel_size=1)

        self.self.bn2 = nn.BatchNorm2d(count_of_1x1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3x3 = nn.Conv2d(count_of_1x1,growth_rate,kernel_size=3)
        
    def forward(self,*prev_features):
        for f in prev_features:
            print(f.shape)

        input = torch.cat(prev_features,dim=1)
        bottleneck_output = self.conv1x1(self.relu1(self.bn1(input)))
        out = self.conv3x3(self.relu2(self.bn1(bottleneck_output)))
        return out



# X=torch.randn(1,10,28,28)
# layer = DenseLayer(10,4,32)
# o=layer(X)
# print(o.shape)


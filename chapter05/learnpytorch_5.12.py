import torch
import torch.nn as nn
import sys
sys.path.append("..") 
import learntorch_utils
import time
import torchvision.models as models

# net = models.DenseNet(growth_rate=32,block_config=(6,12,24,16),num_init_features=64,bn_size=4)
# X = torch.randn((1,3,224,224))
# features = net.features
# features(X)
# for name,module in features.named_children():
#     print(name,module)
#     break

class DenseLayer(nn.Module):
    def __init__(self,in_channels,bottleneck_size,growth_rate):
        super(DenseLayer,self).__init__()
        count_of_1x1 = bottleneck_size
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_channels,count_of_1x1,kernel_size=1)

        self.bn2 = nn.BatchNorm2d(count_of_1x1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3x3 = nn.Conv2d(count_of_1x1,growth_rate,kernel_size=3,padding=1)
        
    def forward(self,*prev_features):
        # for f in prev_features:
        #     print(f.shape)

        input = torch.cat(prev_features,dim=1)
        bottleneck_output = self.conv1x1(self.relu1(self.bn1(input)))
        out = self.conv3x3(self.relu2(self.bn2(bottleneck_output)))
        
        return out


class DenseBlock(nn.Module):
    def __init__(self,in_channels,layer_counts,growth_rate):
        super(DenseBlock,self).__init__()
        self.layer_counts = layer_counts
        self.layers = []
        for i in range(layer_counts):
            curr_input_channel = in_channels + i*growth_rate
            bottleneck_size = 4*growth_rate #论文里设置的1x1卷积核是3x3卷积核的４倍.
            layer = DenseLayer(curr_input_channel,bottleneck_size,growth_rate)        
            self.layers.append(layer)

    def forward(self,init_features):
        features = [init_features]
        for layer in self.layers:
            layer_out = layer(*features) #注意参数是*features不是features
            features.append(layer_out)

        return torch.cat(features, 1)


class TransitionLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self,in_channels,num_classes,block_config):
        super(DenseNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.dense_block_layers = nn.Sequential()

        block_in_channels = in_channels
        growth_rate = 32
        for i,layers_counts in enumerate(block_config):
            block = DenseBlock(in_channels=block_in_channels,layer_counts=layers_counts,growth_rate=growth_rate)
            self.dense_block_layers.add_module('block%d' % (i+1),block)

            block_out_channels = block_in_channels + layers_counts*growth_rate
            transition = TransitionLayer(block_out_channels,block_out_channels//2)
            if i != len(block_config): #最后一个dense block后没有transition layer
                self.dense_block_layers.add_module('transition%d' % (i+1),transition)

            block_in_channels = block_out_channels // 2 #更新下一个dense block的in_channels


    def forward(self,x):
        out = self.conv1(x)
        out = self.pool1(x)
        for layer in self.dense_block_layers:
            out = layer(out) 
            print(out.shape)
        # out = self.block1(out)
        return out

X=torch.randn(1,3,224,224)
block_config = [6,12,24,16]
net = DenseNet(3,10,block_config)
out = net(X)
print(out.shape)

# print(out.shape)

# X=torch.randn(1,10,28,28)
# layer = DenseLayer(10,4,32)
# o=layer(X)
# print(o.shape)


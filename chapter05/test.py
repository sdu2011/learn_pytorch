import torch
import torch.nn as nn
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        print(features)
        self.features = features

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
model = make_layers(cfgs['A'])

X = torch.randn((1,1,224,224))
out = model(X)
#print(out.shape)


# X = torch.randn((16,1,224,224))
# conv1 = nn.Conv2d(1, 6, 5)
# o1 = conv1(X)
# print(o1.shape)
# bn1 = nn.BatchNorm2d(6)
# print(bn1.bias.shape,bn1.weight.shape)
# o2 = bn1(o1)
# print(o2.shape)

def f(a,b):
    b=0.1*b +3
    return a 

b=1
f(2,b)
print(b)

# gamma = torch.ones((1,10,1,1))
# X_hat = torch.ones((256,10,5,5))
# Y = gamma * X_hat #[1,10,1,1]*[256,10,5,5]
# print(Y.shape)
# Y2 = 10 * X_hat
# print(Y2.shape)
# print(Y==Y2)

a=torch.Tensor([[1,2],[3,4]])
b=torch.Tensor([[1,2]])
Y=a*b
print(Y)
Y2 = 2*a
print(Y2)

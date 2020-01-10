# VGG 
AlexNet在Lenet的基础上增加了几个卷积层,改变了卷积核大小,每一层输出通道数目等,并且取得了很好的效果.但是并没有提出一个简单有效的思路.
VGG做到了这一点,提出了可以通过重复使⽤简单的基础块来构建深度学习模型的思路.

论文地址:<https://arxiv.org/abs/1409.1556>

vgg的结构如下所示:
![](https://img2018.cnblogs.com/blog/583030/202001/583030-20200110101932660-2062522034.png)
上图给出了不同层数的vgg的结构.也就是常说的vgg16,vgg19等等.

## VGG BLOCK
vgg的设计思路是,通过不断堆叠3x3的卷积核,不断加深模型深度.vgg net证明了加深模型深度对提高模型的学习能力是一个很有效的手段.

![](https://img2018.cnblogs.com/blog/583030/202001/583030-20200110103318124-1895063989.png)
看上图就能发现,**连续的2个3x3卷积,感受野和一个5x5卷积是一样的,但是前者有两次非线性变换,后者只有一次!**,这就是连续堆叠小卷积核能提高
模型特征学习的关键.此外,2个3x3的参数数量也比一个5x5少.(2x3x3 < 5x5)

vgg的基础组成模块,每一个卷积层都由n个3x3卷积后面接2x2的最大池化.池化层的步幅为2.从而卷积层卷积后,宽高不变,池化后,宽高减半.
我们可以有以下代码:
```
def make_layers(in_channels,cfg):
    layers = []
    previous_channel = in_channels #上一层的输出的channel数量
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            layers.append(nn.Conv2d(previous_channel,v,kernel_size=3,padding=1))
            layers.append(nn.ReLU())

            previous_channel = v

    conv = nn.Sequential(*layers)
    return conv


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
```
cfgs定义了不同的vgg模型的结构,比如'A'代表vgg11．　数字代表卷积后的channel数. 'M'代表Maxpool

我们可以给出模型定义
```
class VGG(nn.Module):
    def __init__(self,input_channels,cfg,num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.conv = make_layers(input_channels,cfg) # torch.Size([1, 512, 7, 7])
        self.fc = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,num_classes)
        )
    
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
```

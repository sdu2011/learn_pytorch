# 批量归一化
论文地址:<https://arxiv.org/abs/1502.03167>  
**批量归一化基本上是现在模型的标配了**.  
说实在的,到今天我也没搞明白batch normalize能够使得模型训练更稳定的底层原因,要彻底搞清楚,涉及到很多凸优化的理论,需要非常扎实的数学基础才行.  
**目前为止,我理解的批量归一化即把每一层输入的特征,统一变换到统一的尺度上来,避免各个特征的单位不统一的情况.即把每一个特征的分布都转变为均值为0,方差为１的分布.
然后在变换后的数据的基础上加一个线性变换.**  
关于batch normalize的常见问题,参考:<https://zhuanlan.zhihu.com/p/55852062>

## 对全连接层做批量归一化
我们先考虑如何对全连接层做批量归一化。通常，我们将批量归一化层置于全连接层中的仿射变换和激活函数之间。设全连接层的输入为$\boldsymbol{u}$，权重参数和偏差参数分别为$\boldsymbol{W}$和$\boldsymbol{b}$，激活函数为$\phi$。设批量归一化的运算符为$\text{BN}$。那么，使用批量归一化的全连接层的输出为

$$\phi(\text{BN}(\boldsymbol{x})),$$

其中批量归一化输入$\boldsymbol{x}$由仿射变换

$$\boldsymbol{x} = \boldsymbol{W\boldsymbol{u} + \boldsymbol{b}}$$

得到。考虑一个由$m$个样本组成的小批量，仿射变换的输出为一个新的小批量$\mathcal{B} = \{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)} \}$。它们正是批量归一化层的输入。对于小批量$\mathcal{B}$中任意样本$\boldsymbol{x}^{(i)} \in \mathbb{R}^d, 1 \leq  i \leq m$，批量归一化层的输出同样是$d$维向量

$$\boldsymbol{y}^{(i)} = \text{BN}(\boldsymbol{x}^{(i)}),$$

并由以下几步求得。首先，对小批量$\mathcal{B}$求均值和方差：

$$\boldsymbol{\mu}_\mathcal{B} \leftarrow \frac{1}{m}\sum_{i = 1}^{m} \boldsymbol{x}^{(i)},$$
$$\boldsymbol{\sigma}_\mathcal{B}^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m}(\boldsymbol{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B})^2,$$

其中的平方计算是按元素求平方。接下来，使用按元素开方和按元素除法对$\boldsymbol{x}^{(i)}$标准化：

$$\hat{\boldsymbol{x}}^{(i)} \leftarrow \frac{\boldsymbol{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},$$

这里$\epsilon > 0$是一个很小的常数，保证分母大于0。在上面标准化的基础上，**批量归一化层引入了两个可以学习的模型参数，拉伸（scale）参数 $\boldsymbol{\gamma}$ 和偏移（shift）参数 $\boldsymbol{\beta}$。这两个参数和$\boldsymbol{x}^{(i)}$形状相同，皆为$d$维向量。这就是文章开头说的对特征做normalization后,再做一次线性变换**
它们与$\boldsymbol{x}^{(i)}$分别做按元素乘法（符号$\odot$）和加法计算：

$${\boldsymbol{y}}^{(i)} \leftarrow \boldsymbol{\gamma} \odot \hat{\boldsymbol{x}}^{(i)} + \boldsymbol{\beta}.$$

至此，我们得到了$\boldsymbol{x}^{(i)}$的批量归一化的输出$\boldsymbol{y}^{(i)}$。
值得注意的是，可学习的拉伸和偏移参数保留了不对$\hat{\boldsymbol{x}}^{(i)}$做批量归一化的可能：此时只需学出$\boldsymbol{\gamma} = \sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}$和$\boldsymbol{\beta} = \boldsymbol{\mu}_\mathcal{B}$。我们可以对此这样理解：如果批量归一化无益，理论上，学出的模型可以不使用批量归一化。

##　对卷积层做批量归一化

对卷积层来说，批量归一化发生在卷积计算之后、应用激活函数之前。如果卷积计算输出多个通道，我们需要对这些通道的输出分别做批量归一化，且**每个通道都拥有独立的拉伸和偏移参数，并均为标量**。设小批量中有$m$个样本。在单个通道上，假设卷积计算输出的高和宽分别为$p$和$q$。我们需要对该通道中$m \times p \times q$个元素同时做批量归一化。对这些元素做标准化计算时，我们使用相同的均值和方差，即该通道中$m \times p \times q$个元素的均值和方差。

用个更具体点的例子总结一下就是:
对于全连接层,假设输出shape为[batch,256],那归一化即对256列的每一列求平均.
对于卷积层,假设输出shape为[batch,96,5,5],即对每个样本来说,有96个5x5的feature map,归一化在96个channel上分别做归一化,均值为batchx5x5个数的均值.

## 预测时的批量归一化
这时候,还有一个问题,就是模型训练好了,传入输入,计算前向传播的结果,也是要做归一化的处理的.那这时候我用的均值和方差应该是多少呢? **很显然,不应该是某个batch的样本的均值和方差,而应该是所有样本的均值和方差.因为gamma和beta的更新是不断累积的结果,而不是仅仅参考某一个batch的输入.(注意,这里的样本不是指模型的输入图片矩阵,而是指归一化层的输入,这个输入随着训练的进行是在不断变化的,而且不同的归一化层的输入是不一样的)**.所以,在做batch normalize的时候,我们还要维护一个值,用于估计全部样本的均值,方差.一种常见的方法是移动平均法.

可以通过下面的测试代码看一下moving_mean是如何逼近3的
```
momentum=0.9
moving_mean = 0.0
for epoch in range(10):
    for mean in [1,2,3,4,5]:
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        print(moving_mean)
```
至于为何不直接对均值之和求平均,我在[torch论坛](https://discuss.pytorch.org/t/what-is-the-running-mean-of-batchnorm-if-gradients-are-accumulated/18870/7)提问了,目前还没回复．

现在我们来总结一下batch normalize的计算过程,然后实现它.分为训练/测试两个部分.  
训练:
- 求输入x的均值
- 求输入x的方差
- 将x归一化
  $$\hat{\boldsymbol{x}}^{(i)} \leftarrow \frac{\boldsymbol{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},$$
- 对归一化后的x做线性变换
  $${\boldsymbol{y}}^{(i)} \leftarrow \boldsymbol{\gamma} \odot \hat{\boldsymbol{x}}^{(i)} + \boldsymbol{\beta}.$$

测试:
- 使用移动平均所得的均值和方差,计算归一化的值
- 对归一化后的值做线性变换

那么可以写出BatchNorm的定义
``` python
def batch_norm(is_training,X,eps,gamma,beta,running_mean,running_var,alpha):
    assert len(X.shape) in (2,4)
    if is_training:
        #X [batch,n]
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X-mean) ** 2).mean(dim=0)
        else:
        #X [batch,c,h,w]
            mean = X.mean(dim=0,keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X-mean) ** 2).mean(dim=0,keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    
        X_hat = (X - mean) / torch.sqrt(var + eps)
        running_mean = alpha * mean + (1 - alpha) * running_mean
        running_var = alpha * var + (1 - alpha) * running_var
    else:
        X_hat = (X - running_mean) / torch.sqrt(running_var + eps)
    
    #print(gamma.shape,X_hat.shape,beta.shape)
    Y = gamma * X_hat + beta  #

    return Y,running_mean,running_var

class BatchNorm(nn.Module):
    def __init__(self,is_conv,in_channels):
        super(BatchNorm,self).__init__()
        #卷积层/全连接层归一化后的线性变换参数.
        if not is_conv:
            # x:[batch,n]
            shape = (1,in_channels)
            self.gamma = nn.Parameter(torch.ones(shape)) #是可学习的参数.反向传播时需要根据梯度更新.
            self.beta = nn.Parameter(torch.zeros(shape)) #是可学习的参数.反向传播时需要根据梯度更新.
            self.running_mean = torch.zeros(shape) #不需要求梯度.在forward时候更新.
            self.running_var = torch.zeros(shape) #不需要求梯度.在forward时候更新.
        else:
            # x:[btach,c,h,w]
            shape = (1,in_channels,1,1)
            self.gamma = nn.Parameter(torch.ones(shape))
            self.beta = nn.Parameter(torch.ones(shape))
            self.running_mean = torch.zeros(shape)
            self.running_var = torch.zeros(shape)

        self.eps = 1e-5
        self.momentum=0.9

    def forward(self,x):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)

        # self.training继承自nn.Module,默认true,调用.eval()会设置成false
        if self.training:
            Y,self.running_mean,self.running_var = batch_norm(True,x,self.eps,self.gamma,self.beta,self.running_mean,self.running_var,self.momentum)
        else:
            Y,self.running_mean,self.running_var = batch_norm(False,x,self.eps,self.gamma,self.beta,self.running_mean,self.running_var,self.momentum)
        
        return Y
```
BatchNorm继承自nn.Module,含有可学习参数`gamma`,`beta`,反向传播时会更新他们. 参数`running_mean`,`running_var`在前向传播时计算.　　　　
`batch_norm`需要区分是卷积后的归一化还是全连接后的归一化.卷积的归一化是对每个channel单独求均值.

## 数据加载
``` python
batch_size,num_workers=16,2
train_iter,test_iter = learntorch_utils.load_data(batch_size,num_workers,None)
```

## 模型定义
``` python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            BatchNorm(is_conv=True,in_channels=6),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            BatchNorm(is_conv=True,in_channels=16),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            BatchNorm(is_conv=False,in_channels=120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(is_conv=False,in_channels = 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = LeNet().cuda()
```

## 损失函数定义
```
l = nn.CrossEntropyLoss()
```

## 优化器定义
```
opt = torch.optim.Adam(net.parameters(),lr=0.01)
```

## 评估函数定义
```
def test():
    acc_sum = 0
    batch = 0
    for X,y in test_iter:
        X,y = X.cuda(),y.cuda()
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        batch += 1
    print('acc:%f' % (acc_sum/(batch*batch_size)))
```

## 训练
```
num_epochs=5
def train():
    for epoch in range(num_epochs):
        train_l_sum,batch=0,0
        start = time.time()
        for X,y in train_iter:
            X,y = X.cuda(),y.cuda() #把tensor放到显存
            y_hat = net(X)  #前向传播
            loss = l(y_hat,y) #计算loss,nn.CrossEntropyLoss中会有softmax的操作
            opt.zero_grad()#梯度清空
            loss.backward()#反向传播,求出梯度
            opt.step()#根据梯度,更新参数

            train_l_sum += loss.item()
            batch += 1
        end = time.time()
        time_per_epoch =  end - start
        print('epoch %d,train_loss %f,time %f' % (epoch + 1,train_l_sum/(batch*batch_size),time_per_epoch))
        test()

train()
```
加了BN层以后,显存直接不够用了.但是用torch自己的`nn.BatchNorm2d`和`nn.BatchNorm1d`就没有问题.应该还是自己的对BatchNorm的实现哪里不够好.

使用torch自己的BatchＮorm的实现定义的模型如下:
``` python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = LeNet().cuda()
```

训练输出如下:
```
epoch 1,batch_size 4,train_loss 0.194394,time 50.538379
acc:0.789400
epoch 2,batch_size 4,train_loss 0.146268,time 52.352518
acc:0.789500
epoch 3,batch_size 4,train_loss 0.132021,time 52.240710
acc:0.820600
epoch 4,batch_size 4,train_loss 0.126241,time 53.277958
acc:0.824400
epoch 5,batch_size 4,train_loss 0.120607,time 52.067259
acc:0.831800

```



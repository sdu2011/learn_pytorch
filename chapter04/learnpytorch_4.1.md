# 模型构造
## nn.Module
nn.Module是pytorch中提供的一个类,是所有神经网络模块的基类.我们自定义的模块要继承这个基类.
```
import torch
from torch import nn

class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层
         

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

X = torch.rand(2, 784)
net = MLP()
print(net)
net(X)
```
输出如下:
```
MLP(
  (hidden): Linear(in_features=784, out_features=256, bias=True)
  (act): ReLU()
  (output): Linear(in_features=256, out_features=10, bias=True)
)
```

## Module的子类
torch中还提供了一些其他的类,方便我们构造模型.**这些类也都是继承自nn.Module.**
- Sequential
- ModuleList
- ModuleDict
- 
这些类的定义都位于torch/nn/modules/container.py

### Sequential
当模型的前向计算为简单串联各个层的计算时，`Sequential`类可以通过更加简单的方式定义模型。这正是`Sequential`类的目的：它可以接收一个子模块的有序字典（OrderedDict）或者一系列子模块作为参数来逐一添加`Module`的实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算。
```
# Example of using Sequential
model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
        )

# Example of using Sequential with OrderedDict
model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1,20,5)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,64,5)),
            ('relu2', nn.ReLU())
        ]))
```

### ModuleList
`ModuleList`接收一个子模块的列表作为输入，然后也可以类似List那样进行append和extend操作:
``` python
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)
# net(torch.zeros(1, 784)) # 会报NotImplementedError
```
输出：
```
Linear(in_features=256, out_features=10, bias=True)
ModuleList(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
```


既然`Sequential`和`ModuleList`都可以进行列表化构造网络，那二者区别是什么呢。**`ModuleList`仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），而且没有实现`forward`功能需要自己实现，所以上面执行`net(torch.zeros(1, 784))`会报`NotImplementedError`；而`Sequential`内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部`forward`功能已经实现。**

`ModuleList`的出现只是让网络定义前向传播时更加灵活，见下面官网的例子。
``` python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
```
这里注意nn.ModuleList传入的是一个python list．
```
nn.ModuleList([nn.Linear(10, 10)])
```
不要写成了
```
nn.ModuleList(nn.Linear(10, 10))
```

另外，**`ModuleList`不同于一般的Python的`list`，加入到`ModuleList`里面的所有模块的参数会被自动添加到整个网络中**，下面看一个例子对比一下。

``` python
class Module_ModuleList(nn.Module):
    def __init__(self):
        super(Module_ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10)])
    
class Module_List(nn.Module):
    def __init__(self):
        super(Module_List, self).__init__()
        self.linears = [nn.Linear(10, 10)]

net1 = Module_ModuleList()
net2 = Module_List()

print("net1:")
for p in net1.parameters():
    print(p.size())

print("net2:")
for p in net2.parameters():
    print(p)
```
输出：
```
net1:
torch.Size([10, 10])
torch.Size([10])
net2:
```
可以看到net2是没有parameters的.net1是有parameters的.因为net1用的是nn.ModuleList而不是python list．

### ＭoduleDict
`ModuleDict`接收一个子模块的字典作为输入, 然后也可以类似字典那样进行添加访问操作:
``` python
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
# net(torch.zeros(1, 784)) # 会报NotImplementedError
```
输出：
```
Linear(in_features=784, out_features=256, bias=True)
Linear(in_features=256, out_features=10, bias=True)
ModuleDict(
  (act): ReLU()
  (linear): Linear(in_features=784, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
```

和`ModuleList`一样，`ModuleDict`实例仅仅是存放了一些模块的字典，并没有定义`forward`函数需要自己定义。同样，`ModuleDict`也与Python的`Dict`有所不同，`ModuleDict`里的所有模块的参数会被自动添加到整个网络中。

**总结一下**
* 可以通过继承`Module`类来构造模型。
* `Sequential`、`ModuleList`、`ModuleDict`类都继承自`Module`类。
* 与`Sequential`不同，`ModuleList`和`ModuleDict`并没有定义一个完整的网络，它们只是将不同的模块存放在一起，需要自己定义`forward`函数。

## 构造复杂模型
上面介绍的Sequential使用简单,但灵活性不足.通常我们还是自定义类,继承nn.Module,去完成更复杂的模型定义和控制.
```
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        
        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)
        
        # 复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))
```
输出
```
FancyMLP(
  (linear): Linear(in_features=20, out_features=20, bias=True)
)
tensor(2.0396, grad_fn=<SumBackward0>)
```
这里在print(net)时的输出,是和__init__函数保持一致的,比如
```
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        
        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)
        
        # 复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))
```
输出
```
FancyMLP(
  (linear): Linear(in_features=20, out_features=20, bias=True)
  (relu): ReLU()
)
tensor(7.5126, grad_fn=<SumBackward0>)
```
尽管在`forward()`里并没有用到`self.relu`.

自定义的模型依然可以和Sequential一起使用.因为再复杂,它也还是继承自nn.Module
```
net = nn.Sequential(nn.Linear(30, 20),FancyMLP())
```




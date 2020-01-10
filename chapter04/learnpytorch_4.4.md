# 自定义layer
<https://www.cnblogs.com/sdu20112013/p/12132786.html>一文里说了怎么写自定义的模型.本篇说怎么自定义层.
分两种:
- 不含模型参数的layer
- 含模型参数的layer

**核心都一样,自定义一个继承自`nn.Module的类`,在类的forward函数里实现该layer的计算,不同的是,带参数的layer需要用到`nn.Parameter`**

## 不含模型参数的layer
直接继承nn.Module
``` python
import torch
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
y.mean().item()
```

## 含模型参数的layer
- Parameter
- ParameterList
- ParameterDict

`Parameter`类其实是`Tensor`的子类，**如果一个`Tensor`是`Parameter`，那么它会自动被添加到模型的参数列表里**。所以在自定义含模型参数的层时，我们应该将参数定义成`Parameter`，除了直接定义成`Parameter`类外，还可以使用`ParameterList`和`ParameterDict`分别定义参数的列表和字典。

ParameterList用法和list类似
```
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense,self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4,4)) for i in range(4)])
        self.params.append(nn.Parameter(torch.randn(4,1)))

    def forward(self,x):
        for i in range(len(self.params)):
            x = torch.mm(x,self.params[i])
        return x

net = MyDense()
print(net)
```
输出
```
MyDense(
  (params): ParameterList(
      (0): Parameter containing: [torch.FloatTensor of size 4x4]
      (1): Parameter containing: [torch.FloatTensor of size 4x4]
      (2): Parameter containing: [torch.FloatTensor of size 4x4]
      (3): Parameter containing: [torch.FloatTensor of size 4x4]
      (4): Parameter containing: [torch.FloatTensor of size 4x1]
  )
)

```

ParameterDict用法和python dict类似.也可以用.keys(),.items()
```
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)

print(net.params.keys(),net.params.items())

x = torch.ones(1, 4)
net(x, 'linear1')
```
输出
```
MyDictDense(
  (params): ParameterDict(
      (linear1): Parameter containing: [torch.FloatTensor of size 4x4]
      (linear2): Parameter containing: [torch.FloatTensor of size 4x1]
      (linear3): Parameter containing: [torch.FloatTensor of size 4x2]
  )
)
odict_keys(['linear1', 'linear2', 'linear3']) odict_items([('linear1', Parameter containing:
tensor([[-0.2275, -1.0434, -1.6733, -1.8101],
        [ 1.7530,  0.0729, -0.2314, -1.9430],
        [-0.1399,  0.7093, -0.4628, -0.2244],
        [-1.6363,  1.2004,  1.4415, -0.1364]], requires_grad=True)), ('linear2', Parameter containing:
tensor([[ 0.5035],
        [-0.0171],
        [-0.8580],
        [-1.1064]], requires_grad=True)), ('linear3', Parameter containing:
tensor([[-1.2078,  0.4364],
        [-0.8203,  1.7443],
        [-1.7759,  2.1744],
        [-0.8799, -0.1479]], requires_grad=True))])
```

使用自定义的layer构造模型
```
layer1 = MyDense()
layer2 = MyDictDense()

net = nn.Sequential(layer2,layer1)
print(net)
print(net(x))
```
输出
```
Sequential(
  (0): MyDictDense(
    (params): ParameterDict(
        (linear1): Parameter containing: [torch.FloatTensor of size 4x4]
        (linear2): Parameter containing: [torch.FloatTensor of size 4x1]
        (linear3): Parameter containing: [torch.FloatTensor of size 4x2]
    )
  )
  (1): MyDense(
    (params): ParameterList(
        (0): Parameter containing: [torch.FloatTensor of size 4x4]
        (1): Parameter containing: [torch.FloatTensor of size 4x4]
        (2): Parameter containing: [torch.FloatTensor of size 4x4]
        (3): Parameter containing: [torch.FloatTensor of size 4x4]
        (4): Parameter containing: [torch.FloatTensor of size 4x1]
    )
  )
)
tensor([[-4.7566]], grad_fn=<MmBackward>)
```
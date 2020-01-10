# GPU计算
默认情况下,pytorch将数据保存在内存,而不是显存.

查看显卡信息
```
nvidia-smi
```
我的机器输出如下:
```
Fri Jan  3 16:20:51 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 105...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   42C    P0    N/A /  N/A |   1670MiB /  4042MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1572      G   /usr/lib/xorg/Xorg                           601MiB |
|    0      4508      G   compiz                                       231MiB |
|    0      4935      G   ...equest-channel-token=592189694510481540   486MiB |
|    0      5574      G   ...quest-channel-token=4527142888685015556   328MiB |
|    0     10049      G   ...passed-by-fd --v8-snapshot-passed-by-fd    21MiB |
+-----------------------------------------------------------------------------+

```
单卡,gtx 1050,4g显存.

查看gpu是否可用
```
torch.cuda.is_available()
```

查看gpu数量
```
torch.cuda.device_count()
```

查看当前gpu号
```
torch.cuda.current_device()
```

查看设备名
```
torch.cuda.get_device_name(device_id)
```

把tensor复制到显存  
使用`.cuda()`可以将CPU上的`Tensor`转换（复制）到GPU上。如果有多块GPU，我们用`.cuda(i)`来表示第 $i$ 块GPU及相应的显存（$i$从0开始）且`cuda(0)`和`cuda()`等价。
``` python
x=x.cuda()
```

直接在显存上存储数据
``` python
device = torch.device('cuda')
x = torch.tensor([1, 2, 3], device=device)
或者
x = torch.tensor([1,2,3]).to(device)
```
如果对在GPU上的数据进行运算，那么结果还是存放在GPU上。
``` python
y = x**2
y
```
输出：
```
tensor([1, 4, 9], device='cuda:0')
```
需要注意的是，存储在不同位置中的数据是不可以直接进行计算的。即存放在CPU上的数据不可以直接与存放在GPU上的数据进行运算，位于不同GPU上的数据也是不能直接进行计算的。
``` python
z = y + x.cpu()
```
会报错:
```
    z=y+x.cpu()
RuntimeError: expected device cuda:0 and dtype Long but got device cpu and dtype Long
```

完整代码
``` python
import torch
from torch import nn

is_gpu = torch.cuda.is_available()
gpu_nums = torch.cuda.device_count()
gpu_index = torch.cuda.current_device()
print(is_gpu,gpu_nums,gpu_index)

device_name = torch.cuda.get_device_name(gpu_index)
print(device_name)

x=torch.Tensor([1,2,3])
print(x)

x=x.cuda(gpu_index)
print(x)

print(x.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3], device=device)
x = torch.tensor([1,2,3]).to(device)
print(x)

y=x**2
print(y)

#z=y+x.cpu()
```

# 模型的gpu计算
同`Tensor`类似，PyTorch模型也可以通过`.cuda`转换到GPU上。我们可以通过检查模型的参数的`device`属性来查看存放模型的设备。

检查模型参数存放设备:
``` python
net = nn.Linear(3,1)
print(type(net.parameters()))
print(list(net.parameters())[0].device)
```
输出
```
<class 'generator'>
cpu
```

在gpu上做运算．通过.cuda()将模型计算放到gpu.相应的,传给模型的输入也必须是gpu显存上的数据.
```
net = nn.Linear(3,1)
print(type(net.parameters()))
print(list(net.parameters())[0].device)

net=net.cuda()
x=torch.tensor([1,2,3]).cuda()
net(x)
```

总结:
* PyTorch可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。在默认情况下，PyTorch会将数据创建在内存，然后利用CPU来计算。
* PyTorch要求计算的所有输入数据都在内存或同一块显卡的显存上。
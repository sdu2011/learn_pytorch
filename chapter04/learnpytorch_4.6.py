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

net = nn.Linear(3,1)
print(type(net.parameters()))
print(list(net.parameters())[0].device)

net=net.cuda()
x=torch.tensor([1,2,3]).cuda()
net(x)




import torch
from torch import nn

x = torch.ones(3)
torch.save(x,'x.pt')

x2 = torch.load('x.pt')
print(x2)

y = torch.zeros(4)
torch.save([x,y],'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)

d = {'x':x,'y':y}
torch.save(d,'xy_dict')

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict())

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

Ｘ=torch.randn(2,3)
Y=net(X)

torch.save(net.state_dict(),"./net.pth")
net2=MLP()
net2.load_state_dict(torch.load("./net.pth"))
Y2=net2(X)
#print(Y,Y2)
print(Y==Y2)




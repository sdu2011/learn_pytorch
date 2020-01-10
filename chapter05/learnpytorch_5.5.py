import torch
from torch import nn
import sys
sys.path.append("..") 
import learntorch_utils

batch_size,num_workers=64,4
train_iter,test_iter = learntorch_utils.load_data(batch_size,num_workers,None)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

device = torch.device('cuda')
l = nn.CrossEntropyLoss()
net = LeNet().cuda()
opt = torch.optim.Adam(net.parameters(),lr=0.01)

num_epochs=5
def test():
    acc_sum = 0
    batch = 0
    for X,y in test_iter:
        X,y = X.cuda(),y.cuda()
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        batch += 1
    print('acc:%f' % (acc_sum/(batch*batch_size)))

def train():
    for epoch in range(num_epochs):
        train_l_sum,batch=0,0
        for X,y in train_iter:
            X,y = X.cuda(),y.cuda() #把tensor放到显存
            y_hat = net(X)  #前向传播
            loss = l(y_hat,y) #计算loss,nn.CrossEntropyLoss中会有softmax的操作
            opt.zero_grad()#梯度清空
            loss.backward()#反向传播,求出梯度
            opt.step()#根据梯度,更新参数

            train_l_sum += loss.item()
            batch += 1
        print('epoch %d,train_loss %f' % (epoch + 1,train_l_sum/(batch*batch_size)))
        test()

train()

net0 = nn.Sequential(
        nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2), # kernel_size, stride
        nn.Conv2d(6, 16, 5),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2)
    )

X = torch.randn((batch_size,1,28,28))
out=net0(X)
#print(out.shape)

#test()





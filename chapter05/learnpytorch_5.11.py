import torch
import torch.nn as nn
import sys
sys.path.append("..") 
import learntorch_utils
import time
import torchvision.models as models

models.resnet

class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(Residual,self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #　ｘ卷积后shape发生改变,比如:x:[1,64,56,56] --> [1,128,28,28],则需要1x1卷积改变x
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv1x1 = None
            

    def forward(self,x):
        # print(x.shape)
        o1 = self.relu(self.bn1(self.conv1(x)))
        # print(o1.shape)
        o2 = self.bn2(self.conv2(o1))
        # print(o2.shape)

        if self.conv1x1:
            x = self.conv1x1(x) 

        out = self.relu(o2 + x)
        return out

class ResNet(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(ResNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            Residual(64,64),
            Residual(64,64),
            Residual(64,64),
        )

        self.conv3 = nn.Sequential(
            Residual(64,128,stride=2),
            Residual(128,128),
            Residual(128,128),
            Residual(128,128),
            Residual(128,128),
        )

        self.conv4 = nn.Sequential(
            Residual(128,256,stride=2),
            Residual(256,256),
            Residual(256,256),
            Residual(256,256),
            Residual(256,256),
            Residual(256,256),
        )

        self.conv5 = nn.Sequential(
            Residual(256,512,stride=2),
            Residual(512,512),
            Residual(512,512),
        )

        # self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #代替AvgPool2d以适应不同size的输入
        self.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        
        out = self.avg_pool(out)
        out = out.view((x.shape[0],-1))

        out = self.fc(out)

        return out

# net = ResNet(1,10)
# X=torch.randn((1,1,128,128))
# out = net(X)
# print(out.shape)

## 数据加载
batch_size,num_workers=32,2
train_iter,test_iter = learntorch_utils.load_data(batch_size,num_workers,resize=128)
print('load data done,batch_size:%d' % batch_size)

## 模型定义
net = ResNet(1,10).cuda()

## 损失函数定义
l = nn.CrossEntropyLoss()

## 优化器定义
opt = torch.optim.Adam(net.parameters(),lr=0.01)

## 评估函数定义
num_epochs=5
def test():
    acc_sum = 0
    batch = 0
    for X,y in test_iter:
        X,y = X.cuda(),y.cuda()
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        batch += 1
    
    test_acc = acc_sum/(batch*batch_size)

    # print('test acc:%f' % test_acc)
    return test_acc

## 训练
def train():
    for epoch in range(num_epochs):
        train_l_sum,batch,train_acc_sum=0,1,0
        start = time.time()
        for X,y in train_iter:
            X,y = X.cuda(),y.cuda() #把tensor放到显存
            y_hat = net(X)  #前向传播
            loss = l(y_hat,y) #计算loss,nn.CrossEntropyLoss中会有softmax的操作
            opt.zero_grad()#梯度清空
            loss.backward()#反向传播,求出梯度
            opt.step()#根据梯度,更新参数

            #　数据统计
            train_l_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            train_loss = train_l_sum/(batch*batch_size)
            train_acc = train_acc_sum/(batch*batch_size)
            
            if batch % 100 == 0: #每100个batch输出一次训练数据
                print('epoch %d,batch %d,train_loss %.3f,train_acc:%.3f' % (epoch,batch,train_loss,train_acc))

            if batch % 300 == 0: #每300个batch测试一次
                test_acc = test()
                print('epoch %d,batch %d,test_acc:%.3f' % (epoch,batch,test_acc))

            batch += 1

        end = time.time()
        time_per_epoch =  end - start
        print('epoch %d,batch_size %d,train_loss %f,time %f' % 
                (epoch + 1,batch_size ,train_l_sum/(batch*batch_size),time_per_epoch))
        test()

train()


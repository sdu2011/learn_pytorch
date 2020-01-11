#https://www.cnblogs.com/sdu20112013/p/12176304.html

import torchvision.models as models
import torch
from torch import nn
import sys
sys.path.append("..")
import learntorch_utils
import time

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

# class VGG(nn.Module):
#     def __init__(self,input_channels,cfg,num_classes=10, init_weights=True):
#         super(VGG, self).__init__()
#         self.conv = make_layers(input_channels,cfg) # torch.Size([1, 512, 7, 7])
#         self.fc = nn.Sequential(
#             nn.Linear(512*7*7,4096),
#             nn.ReLU(inplace=True), #inplace作用:节省显存　https://www.cnblogs.com/wanghui-garcia/p/10642665.html
#             nn.Dropout(p=0.5),
#             nn.Linear(4096,4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096,num_classes)
#         )
    
#     def forward(self, img):
#         feature = self.conv(img)
#         output = self.fc(feature.view(img.shape[0], -1))
#         return output

class VGG(nn.Module):
    def __init__(self,input_channels,cfg,num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.conv = make_layers(input_channels,cfg) # torch.Size([1, 512, 7, 7])
        self.fc = nn.Sequential(
            nn.Linear(512*7*7,512),
            nn.ReLU(inplace=True), #inplace作用:节省显存　https://www.cnblogs.com/wanghui-garcia/p/10642665.html
            nn.Dropout(p=0.5),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,num_classes)
        )
    
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

# conv = make_layers(1,cfgs['A'])
# X = torch.randn((1,1,224,224))
# out = conv(X)
# #print(out.shape)

# net = VGG(1,cfgs['A'])
# out = net(X)
# print(out.shape)

# 加载数据
batch_size,num_workers=16,4
train_iter,test_iter = learntorch_utils.load_data(batch_size,num_workers,resize=224)

# 定义模型
net = VGG(1,cfgs['A']).cuda()
print(net)

#　恢复模型参数
net.load_state_dict(torch.load('./vgg_epoch_0_batch_2000_acc_0.81.pt'))

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 定义优化器　
opt = torch.optim.Adam(net.parameters(),lr=0.001)

# 定义评估函数
def test():
    start = time.time()
    acc_sum = 0
    batch = 0
    for X,y in test_iter:
        X,y = X.cuda(),y.cuda()
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        batch += 1
    #print('acc_sum %d,batch %d' % (acc_sum,batch))
    
    acc = 1.0*acc_sum/(batch*batch_size)
    end = time.time()
    print('acc %3f,test for test dataset:time %d' % (acc,end - start))

    return acc

# 训练
num_epochs = 3
def train():
    for epoch in range(num_epochs):
        train_l_sum,batch,acc_sum = 0,0,0
        start = time.time()
        for X,y in train_iter:
            # start_batch_begin = time.time()
            X,y = X.cuda(),y.cuda()
            y_hat = net(X)
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()

            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()

            opt.step()
            train_l_sum += l.item()

            batch += 1

            mean_loss = train_l_sum/(batch*batch_size) #计算平均到每张图片的loss
            start_batch_end = time.time()
            time_batch = start_batch_end - start

            train_acc = acc_sum/(batch*batch_size) 
            if batch % 100 == 0:
                print('epoch %d,batch %d,train_loss %.3f,train_acc:%.3f,time %.3f' % 
                    (epoch,batch,mean_loss,train_acc,time_batch))

            if batch % 1000 == 0:
                model_state = net.state_dict()
                model_name = 'vgg_epoch_%d_batch_%d_acc_%.2f.pt' % (epoch,batch,train_acc)
                torch.save(model_state,model_name)

        print('***************************************')
        mean_loss = train_l_sum/(batch*batch_size) #计算平均到每张图片的loss
        train_acc = acc_sum/(batch*batch_size)     #计算训练准确率
        test_acc = test()                           #计算测试准确率
        end = time.time()
        time_per_epoch =  end - start
        print('epoch %d,train_loss %f,train_acc %f,test_acc %f,time %f' % 
                (epoch + 1,mean_loss,train_acc,test_acc,time_per_epoch))

train()
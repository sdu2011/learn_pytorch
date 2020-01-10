import torchvision.models.alexnet
import torch
from torch import nn
import sys
sys.path.append("..")
import learntorch_utils
import time

# #net = nn.Conv2d(1, 96, 11, 4,4)
# #X= torch.randn((1,1,224,224))
# # out=net(X)
# # print(out.shape)

# net = nn.Conv2d(1, 1, 3, padding=0)
# X = torch.randn((1, 1, 5, 5))
# # print(X)
# out = net(X)
# # print(out)

# net2 = nn.Conv2d(1, 1, 3, padding=1)
# out = net(X)
# # print(out)

# X = torch.randn((1, 1, 224, 224))  # batch x c x h x w
# net = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2)
# out = net(X)
# # print(out.shape)



# # class AlexNet(nn.Module):
# #     def __init__(self):
# #         super(AlexNet, self).__init__()
# #         self.conv = nn.Sequential(
# #             nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
# #             nn.ReLU(),
# #             nn.MaxPool2d(3, 2), # kernel_size, stride
# #             # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
# #             nn.Conv2d(96, 256, 5, 1, 2),
# #             nn.ReLU(),
# #             nn.MaxPool2d(3, 2),
# #             # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
# #             # 前两个卷积层后不使用池化层来减小输入的高和宽
# #             nn.Conv2d(256, 384, 3, 1, 1),
# #             nn.ReLU(),
# #             nn.Conv2d(384, 384, 3, 1, 1),
# #             nn.ReLU(),
# #             nn.Conv2d(384, 256, 3, 1, 1),
# #             nn.ReLU(),
# #             nn.MaxPool2d(3, 2)
# #         )

# # AlexNet有5个卷积层,第一个卷积层的卷积核大小为11x11,第二个卷积层的卷积核大小为5x5,其余的卷积核均为3x3.第一二五个卷积层后做了最大池化操作,窗口大小为3x3,步幅为2.
# net = nn.Sequential(
#     # [1,96,55,55] order:batch,channel,h,w
#     nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2),  # [1,96,27,27]

#     nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # [1,256,27,27]
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2),  # [1,256,13,13]

#     nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # [1,384,13,13]
#     nn.ReLU(),

#     nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # [1,384,13,13]
#     nn.ReLU(),

#     nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # [1,256,13,13]
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2),  # [1,256,6,6]
# )
# out = net(X)
# print(out.shape)

# fc_input = torch.randn((1, 256, 6, 6))  # batch x c x h x w
# net = nn.Sequential(
#     nn.Linear(256*6*6, 4096),
#     nn.ReLU(),
#     nn.Dropout(0.5),  # dropout防止过拟合
#     nn.Linear(4096, 4096),
#     nn.ReLU(),
#     nn.Dropout(0.5),  # dropout防止过拟合
#     nn.Linear(4096, 10)  # 我们最终要10分类
# )

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(  
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2), # [1,96,55,55] order:batch,channel,h,w
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [1,96,27,27]

            nn.Conv2d(96, 256, kernel_size=5, stride=1,padding=2),  # [1,256,27,27]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [1,256,13,13]

            nn.Conv2d(256, 384, kernel_size=3, stride=1,padding=1),  # [1,384,13,13]
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, stride=1,padding=1),  # [1,384,13,13]
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, stride=1,padding=1),  # [1,256,13,13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [1,256,6,6]
        )

        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),  # dropout防止过拟合
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),  # dropout防止过拟合
            nn.Linear(4096, 10)  # 我们最终要10分类
        )
    
    def forward(self, img):
        feature = self.conv(img) 
        output = self.fc(feature.view(img.shape[0], -1))#输入全连接层之前,将特征展平
        return output

# 加载数据
batch_size,num_workers=128,4
train_iter,test_iter = learntorch_utils.load_data(batch_size,num_workers,resize=224)

# 定义模型
net = AlexNet().cuda()

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 定义优化器　
opt = torch.optim.Adam(net.parameters(),lr=0.001)

# 定义评估函数
def test():
    acc_sum = 0
    batch = 0
    for X,y in test_iter:
        X,y = X.cuda(),y.cuda()
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        batch += 1
    #print('acc_sum %d,batch %d' % (acc_sum,batch))

    return 1.0*acc_sum/(batch*batch_size)

# 训练
num_epochs = 3
def train():
    for epoch in range(num_epochs):
        train_l_sum,batch,acc_sum = 0,0,0
        start = time.time()
        for X,y in train_iter:
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
        train_acc = acc_sum/(batch*batch_size)     #计算训练准确率
        test_acc = test()                           #计算测试准确率
        end = time.time()
        time_per_epoch =  end - start
        print('epoch %d,train_loss %f,train_acc %f,test_acc %f,time %f' % 
                (epoch + 1,mean_loss,train_acc,test_acc,time_per_epoch))

train()
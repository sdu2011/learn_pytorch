import torch
import torch.nn as nn
import sys
sys.path.append("..") 
import learntorch_utils
import time

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

        
# X=torch.randn((3,1,2,2))
# X_copy = X
# print(X)
# mean = X.mean(dim=0,keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
# print(mean)

# mean_copy = X_copy.mean(dim=0,keepdim=True).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
# print(mean_copy)

# def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
#     # 判断当前模式是训练模式还是预测模式
#     if not is_training:
#         # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
#         X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
#     else:
#         assert len(X.shape) in (2, 4)
#         if len(X.shape) == 2:
#             # 使用全连接层的情况，计算特征维上的均值和方差
#             mean = X.mean(dim=0)
#             var = ((X - mean) ** 2).mean(dim=0)
#         else:
#             # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
#             # X的形状以便后面可以做广播运算
#             mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
#             var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
#         # 训练模式下用当前的均值和方差做标准化
#         X_hat = (X - mean) / torch.sqrt(var + eps)
#         # 更新移动平均的均值和方差
#         moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
#         moving_var = momentum * moving_var + (1.0 - momentum) * var
#     Y = gamma * X_hat + beta  # 拉伸和偏移
#     return Y, moving_mean, moving_var

# class BatchNorm(nn.Module):
#     def __init__(self, num_features, num_dims):
#         super(BatchNorm, self).__init__()
#         if num_dims == 2:
#             shape = (1, num_features)
#         else:
#             shape = (1, num_features, 1, 1)
#         # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
#         self.gamma = nn.Parameter(torch.ones(shape))
#         self.beta = nn.Parameter(torch.zeros(shape))
#         # 不参与求梯度和迭代的变量，全在内存上初始化成0
#         self.moving_mean = torch.zeros(shape)
#         self.moving_var = torch.zeros(shape)

#     def forward(self, X):
#         # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
#         if self.moving_mean.device != X.device:
#             self.moving_mean = self.moving_mean.to(X.device)
#             self.moving_var = self.moving_var.to(X.device)
#         # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
#         Y, self.moving_mean, self.moving_var = batch_norm(self.training, 
#             X, self.gamma, self.beta, self.moving_mean,
#             self.moving_var, eps=1e-5, momentum=0.9)
#         return Y

## 数据加载
batch_size,num_workers=4,2
train_iter,test_iter = learntorch_utils.load_data(batch_size,num_workers,None)

## 模型定义
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
    print('acc:%f' % (acc_sum/(batch*batch_size)))

## 训练
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
        print('epoch %d,batch_size %d,train_loss %f,time %f' % 
                (epoch + 1,batch_size ,train_l_sum/(batch*batch_size),time_per_epoch))
        test()

train()

# momentum=0.9
# moving_mean = 0.0
# for epoch in range(40):
#     for mean in [1,2,3,4,5]:
#         moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
#         print(moving_mean)


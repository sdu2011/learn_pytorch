import torch
from torch import nn
import sys
sys.path.append("..")
import learntorch_utils
import time

# def nin_block(in_channels, out_channels, kernel_size, stride, padding):
#     blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
#                         nn.ReLU(),
#                         nn.Conv2d(out_channels, out_channels, kernel_size=1),
#                         nn.ReLU(),
#                         nn.Conv2d(out_channels, out_channels, kernel_size=1),
#                         nn.ReLU())
#     return blk


def make_layers(in_channels,out_channels,kernel_size, stride, padding):
    conv = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,kernel_size=1, stride=1, padding=0),#1x1卷积,整合多个feature map的特征
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,kernel_size=1, stride=1, padding=0),#1x1卷积,整合多个feature map的特征
        nn.ReLU(inplace=True)
    )

    return conv

# conv1 = make_layers(1,96,11,4,2)
# pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
# conv2 = make_layers(96,256,kernel_size=5,stride=1,padding=2)
# pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
# conv3 = make_layers(256,384,kernel_size=3,stride=1,padding=1) 
# pool3 = nn.MaxPool2d(kernel_size=3,stride=2)
# conv4 = make_layers(384,10,kernel_size=3,stride=1,padding=1) 

# X = torch.rand(1, 1, 224, 224)
# o1 = conv1(X) 
# print(o1.shape) #[1,96,55,55]
# o1_1 = pool1(o1)
# print(o1_1.shape) #[1,96,27,27]

# o2 = conv2(o1_1) 
# print(o2.shape) #[1,256,27,27]
# o2_1 = pool2(o2)
# print(o2_1.shape) #[1,256,13,13]

# o3 = conv3(o2_1) 
# print(o3.shape) #[1,384,13,13]
# o3_1 = pool3(o3)
# print(o3_1.shape) #[1,384,6,6]

# o4 = conv4(o3_1)
# print(o4.shape) #[1,10,6,6]

# ap = nn.AvgPool2d(kernel_size=6,stride=1)
# o5 = ap(o4)
# print(o5.shape) #[1,10,1,1]

class ＮinNet(nn.Module):
    def __init__(self):
        super(ＮinNet, self).__init__()
        self.conv = nn.Sequential(  
            make_layers(1,96,11,4,2),
            nn.MaxPool2d(kernel_size=3,stride=2),
            make_layers(96,256,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=3,stride=2),
            make_layers(256,384,kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            make_layers(384,10,kernel_size=3,stride=1,padding=1) 
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6,stride=1)
        )
    
    def forward(self, img):
        feature = self.conv(img) 
        output = self.gap(feature)
        output = output.view(img.shape[0],-1) #[batch,10,1,1]-->[batch,10]

        return output

X = torch.rand(1, 1, 224, 224)
net = ＮinNet()
for name,module in net.named_children():
    X = module(X)
    print(name,X.shape)


# 加载数据
batch_size,num_workers=16,4
train_iter,test_iter = learntorch_utils.load_data(batch_size,num_workers,resize=224)

# 定义模型
net = ＮinNet().cuda()
print(net)

# #　恢复模型参数
# net.load_state_dict(torch.load('./vgg_epoch_0_batch_2000_acc_0.81.pt'))

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
save_to_disk = False
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

            if save_to_disk and batch % 1000 == 0:
                model_state = net.state_dict()
                model_name = 'nin_epoch_%d_batch_%d_acc_%.2f.pt' % (epoch,batch,train_acc)
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
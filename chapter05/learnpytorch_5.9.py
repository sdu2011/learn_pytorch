import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys
sys.path.append("..")
import learntorch_utils
import time

class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_c,c1,kernel_size=1)
        
        self.branch2_1 = nn.Conv2d(in_c,c2[0],kernel_size=1)
        self.branch2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)

        self.branch3_1 = nn.Conv2d(in_c,c3[0],kernel_size=1)
        self.branch3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)

        self.branch4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = nn.Conv2d(in_c,c4,kernel_size=1)

    def forward(self,x):
        o1 = self.branch1(x)
        o1 = F.relu(o1)
        # print("o1:",o1.shape)
        
        o2 = self.branch2_1(x)
        o2 = F.relu(o2)
        o2 = self.branch2_2(o2)
        o2 = F.relu(o2)
        # print("o2:",o2.shape)

        o3 = self.branch3_1(x)
        o3 = F.relu(o3)
        o3 = self.branch3_2(o3)
        o3 = F.relu(o3)
        # print("o3:",o3.shape)

        o4 = self.branch4_1(x)
        o4 = self.branch4_2(o4)
        o4 = F.relu(o4)
        # print("o4:",o4.shape)

        concat = torch.cat((o1,o2,o3,o4),dim=1)
        # print("concat:",concat.shape)

        return concat

# X = torch.randn((1,1,224,224))
# conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3)
# max_pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
# o=conv1(X)
# print(o.shape) #[1,64,112,112]
# o=max_pool1(o)
# print(o.shape) #[1,64,56,56]

# conv2_1 = nn.Conv2d(64,64,kernel_size=1)
# conv2_2 = nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1)
# max_pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
# o=conv2_1(o)
# print(o.shape) #[1,64,56,56]
# o=conv2_2(o)
# print(o.shape) #[1,192,56,56]
# o=max_pool2(o)
# print(o.shape) #[1,192,28,28]

# inception_3a = Inception(192,64,(96,128),(16,32),32)
# o=inception_3a(o)
# print(o.shape)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,192,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )

        self.inception_3a = Inception(192,64,(96,128),(16,32),32)
        self.inception_3b = Inception(256,128,(128,192),(32,96),64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception_4a = Inception(480,192,(96,208),(16,48),64)
        self.inception_4b = Inception(512,160,(112,224),(24,64),64)
        self.inception_4c = Inception(512,128,(128,256),(24,64),64)
        self.inception_4d = Inception(512,112,(144,288),(32,64),64)
        self.inception_4e = Inception(528,256,(160,320),(32,128),128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception_5a = Inception(832,256,(160,320),(32,128),128)
        self.inception_5b = Inception(832,384,(192,384),(48,128),128)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.dropout = nn.Dropout(p=0.4,inplace=True)

        self.fc = nn.Linear(1024,10,bias=True)

    def forward(self,x):
        feature = self.conv1(x)
        feature = self.conv2(feature)
        
        feature = self.inception_3a(feature)
        feature = self.inception_3b(feature)
        feature = self.max_pool3(feature)
        
        feature = self.inception_4a(feature)
        feature = self.inception_4b(feature)
        feature = self.inception_4c(feature)
        feature = self.inception_4d(feature)
        feature = self.inception_4e(feature)
        feature = self.max_pool4(feature)

        feature = self.inception_5a(feature)
        feature = self.inception_5b(feature)

        feature = self.avg_pool(feature)
        feature = self.dropout(feature)

        out = self.fc(feature.view(x.shape[0],-1))

        return out


# X=torch.randn((1,1,224,224))
# net = GoogLeNet()
# for name,module in net.named_children():
#     X=module(X)
#     print(name,X.shape) 

# out = net(X)
# print(out.shape)

# 加载数据
batch_size,num_workers=32,4
train_iter,test_iter = learntorch_utils.load_data(batch_size,num_workers,resize=224)

# 定义模型
net = GoogLeNet().cuda()
print(net)

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
            if batch % 100 == 0: #每100个batch输出一次训练数据
                print('epoch %d,batch %d,train_loss %.3f,train_acc:%.3f,time %.3f' % 
                    (epoch,batch,mean_loss,train_acc,time_batch))

            if batch % 300 == 0: #每300个batch测试一次
                test_acc = test()
                print('epoch %d,batch %d,test_acc:%.3f' % 
                    (epoch,batch,test_acc))

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

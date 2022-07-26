'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-07-28 19:08:07
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-07-31 10:44:25
FilePath: \\NTUAILab\\CervicalCancerRiskClassification\\train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from torchvision.models import alexnet
from torchvision.models import vgg16
from torchvision.models import resnet50
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from createDataLoader import LoadData
from torch.utils.tensorboard import SummaryWriter # Load SummaryWriter
# AlexNet 150epoch; Accuracy: 57.8%
from hqNet import AlexNet
from torch_deform_conv.cnn import get_cnn, get_deform_cnn

# writer = SummaryWriter("logs") # put file into logs folder

# train_root  = "E:\\NTUAILab\\Data\\224_224_CervicalCancerScreening\\kaggle\\train\\train"
train_root  = "/data/hyang/224_224_CervicalCancerScreening/kaggle/train/train/"
# test_root = "E:\\NTUAILab\\Data\\224_224_CervicalCancerScreening\\kaggle\\test\\test"
test_root = "/data/hyang/224_224_CervicalCancerScreening/kaggle/test/test"

#if wanted to display image 
# img = Image.open('E:\\NTUAILab\\Data\\224_224_CervicalCancerScreening\\kaggle\\train\\train\\Type_1\\0.jpg')
# plt.imshow(img)
# print("Example Image Size: {}".format(img.size))
# image_path="E:\\NTUAILab\\Data\\224_224_CervicalCancerScreening\\kaggle\\train\\train\\Type_1\\0.jpg"
# image_path="/data/hyang/224_224_CervicalCancerScreening/kaggle/train/train/Type_1/0.jpg"

# img=Image.open(image_path) # PIL库中的Image来打开指定路径下的图片，并将数据存入imgzhong
# img_arrey = np.array(img) # 数据类型转化
# writer.add_image("example_img",img_arrey,2,dataformats='HWC') #HWC类型格式，即高、宽、通道RBG

# Set the currently used GPU device to device 0 only
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   

# Define the device, and whether to use GPU will be automatically selected according to the computer configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# num_classes = len([i for i in os.listdir(train_root)])
# print("Number of Classes: {}".format(num_classes))


# 定义训练函数，需要
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        # print("batch: {}".format(batch))
        # 将数据存到显卡
        X, y = X.cuda(), y.cuda()
 
        # 得到预测的结果pred
        pred = model(X)
 
        # 计算预测的误差
        # print(pred,y)
        loss = loss_fn(pred, y)
 
        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        # # 每训练100次，输出一次当前信息
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
 
 
def test(dataloader, model):
    size = len(dataloader.dataset)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            # 将数据转到GPU
            X, y = X.cuda(), y.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print("correct = ",correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
 
 
if __name__=='__main__':
    batch_size = 3
 
    # # 给训练集和测试集分别创建一个数据集加载器
    train_data = LoadData("train.txt", True)
    valid_data = LoadData("test.txt", False)
 
 
    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size)
 
    for X, y in train_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break
 
 
    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
 
    # 调用刚定义的模型，将模型转到GPU（如果可用）
    '''
        1. AlexNet
    '''
    # model = AlexNet().to(device)
    # model = alexnet(pretrained=True)
    # model.classifier[6] = nn.Linear(4096, 3)
    '''
        2. VGG16
    '''
    # model = vgg16(pretrained=True)
    # model.classifier[6] = nn.Linear(4096, 3)
    '''
        3. ResNet50
    '''
    # model = resnet50(pretrained=True)
    # model.fc = nn.Linear(2048, 3)
    '''
        4. Deformable Conv
    '''
    model = get_deform_cnn(trainable=True)
    model.fc = nn.Linear(128, 3)
    
    model.to(device)
    print(model)
 
    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()
 
    # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 初始学习率
    '''
        Optimization 1: Change optimizer from SGD to Nadam
    '''
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3)  # 初始学习率


    # 一共训练1500次
    epochs = 1500
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)

    print("Done!")
 
    # 保存训练好的模型
    # torch.save(model.state_dict(), "E:\\NTUAILab\\CervicalCancerRiskClassification\\exp\\DConvmodel.pth")
    torch.save(model.state_dict(), "exp/DConvmodel.pth")
    print("Saved PyTorch Model State to exp/DConvmodel.pth")
 
 
    # 读取训练好的模型，加载训练好的参数
    # model = NeuralNetwork()
    # model.load_state_dict(torch.load("model.pth"))
 
 
    # # 定义所有类别
    # classes = [
    #     "Type1",
    #     "Type2",
    #     "Type3",
    # ]
    
    # # 模型进入验证阶段
    # model.eval()
    
    # x, y = test_data[0][0], test_data[0][1]
    # with torch.no_grad():
    #     pred = model(x)
    #     predicted, actual = classes[pred[0].argmax(0)], classes[y]
    #     print(f'Predicted: "{predicted}", Actual: "{actual}"')

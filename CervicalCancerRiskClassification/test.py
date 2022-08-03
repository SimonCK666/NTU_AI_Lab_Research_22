'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-07-29 16:17:22
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-08-03 10:33:26
FilePath: \\NTUAILab\\CervicalCancerRiskClassification\\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch import nn
from torchvision.models import alexnet
from torchvision.models import vgg16
from torchvision.models import vgg19
from torchvision.models import resnet50
from createDataLoader import LoadData
from torch.utils.data import Dataset, DataLoader
from vit_pytorch.deepvit import DeepViT
from hqNet import HQNet


def model():
    # alexnetmodel = alexnet()
    # alexnetmodel.classifier[6] = nn.Linear(4096, 3)
    # print(alexnetmodel)

    # vgg16model = vgg16()
    # vgg16model.classifier[6] = nn.Linear(4096, 3)
    # print(vgg16model)
    
    # resnet50model = resnet50()
    # resnet50model.fc = nn.Linear(2048, 3)
    # print(resnet50model)
    
    vgg19model = vgg19()
    vgg19model.classifier[6] = nn.Linear(4096, 5)
    print(vgg19model)
    

def createViT():
    deepViT = DeepViT(
        image_size = 224,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    print(deepViT)
    

def train(dataloader):
    size = len(dataloader.dataset)
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        print(batch)


def data():
    batch_size = 14
    # # 给训练集和测试集分别创建一个数据集加载器
    train_data = LoadData("train.txt", True)
    valid_data = LoadData("test.txt", False)
 
 
    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size)
 
    # for X, y in train_dataloader:
    #     print("Shape of X [N, C, H, W]: ", X.shape)
    #     print("Shape of y: ", y.shape, y.dtype)
        # break
    
    # for batch, (X, y) in enumerate(train_dataloader):
    #     print("batch: ", batch)
    #     # print("data: ", data)
    #     # break
    
    train(train_dataloader)


def createHQ():
    hqmodel = HQNet(pretrained=False)
    print(hqmodel)


def testDropout():
    dmodel = nn.Dropout2d(p=0.2)
    input = torch.randn(20, 16, 32, 32)
    out = dmodel(input)
    print(out.shape)


def main():
    model()
    # data()
    # createViT()
    # createHQ()
    # testDropout()

if __name__ == "__main__":
    main()
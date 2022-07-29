'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-07-29 16:17:22
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-07-29 16:19:33
FilePath: \\NTUAILab\\CervicalCancerRiskClassification\\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torch import nn
from torchvision.models import alexnet
from torchvision.models import vgg16
from torchvision.models import resnet50


def main():
    alexnetmodel = alexnet()
    alexnetmodel.classifier[6] = nn.Linear(4096, 3)
    print(alexnetmodel)

    vgg16model = vgg16()
    vgg16model.classifier[6] = nn.Linear(4096, 3)
    print(vgg16model)
    
    resnet50model = resnet50()
    resnet50model.fc = nn.Linear(2048, 3)
    print(resnet50model)

if __name__ == "__main__":
    main()
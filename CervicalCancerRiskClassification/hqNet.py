'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-07-28 19:05:28
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-07-31 10:20:42
FilePath: \\NTUAILab\\CervicalCancerRiskClassification\\hqNet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from operator import mod
from torch import nn, optim
import torch
from torchvision.ops import deform_conv2d

'''
    Deformable Conv
    torchvision.ops.deform_conv2d(input: Tensor, offset: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Tuple[int, int] = (1, 1), 
                                  padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), mask: Optional[Tensor] = None) → Tensor
    input (Tensor[batch_size, in_channels, in_height, in_width]) - input tensor

    offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]) - offsets to be applied for each position in the convolution kernel.

    weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]) - convolution weights, split into groups of size (in_channels // groups)

    bias (Tensor[out_channels]) - optional bias of shape (out_channels,). Default: None

    stride (int or Tuple[int, int]) - distance between convolution centers. Default: 1

    padding (int or Tuple[int, int]) - height/width of padding of zeroes around each image. Default: 0

    dilation (int or Tuple[int, int]) - the spacing between kernel elements. Default: 1

    mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]) - masks to be applied for each position in the convolution kernel. Default: None

    Using
    >>> input = torch.rand(4, 3, 10, 10)
    >>> kh, kw = 3, 3
    >>> weight = torch.rand(5, 3, kh, kw)
    >>> # offset and mask should have the same spatial size as the output
    >>> # of the convolution. In this case, for an input of 10, stride of 1
    >>> # and kernel size of 3, without padding, the output size is 8
    >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
    >>> mask = torch.rand(4, kh * kw, 8, 8)
    >>> out = deform_conv2d(input, offset, weight, mask=mask)
'''

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-07-28 19:26:40
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-07-28 19:26:42
FilePath: \\NTUAILab\\CervicalCancerRiskClassification\\createDataLoader.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
    createDataloader.py 
'''
import torch
from PIL import Image
 
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
from torch.utils.data import Dataset
 
# Data normalization and standardization
 
# Image standardization
transform_BZ= transforms.Normalize(
    mean=[0.5, 0.5, 0.5],# 取决于数据集
    std=[0.5, 0.5, 0.5]
)
 
 
class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag
 
        self.train_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transform_BZ
            ])
        self.val_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transform_BZ
            ])
 
    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info
 
    def padding_black(self, img):
        w, h  = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img
 
    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
 
        return img, label
 
    def __len__(self):
        return len(self.imgs_info)
 
 
if __name__ == "__main__":
    train_dataset = LoadData("train.txt", True)
    print("Number of Data: ", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(image)
        # img = transform_BZ(image)
        # print(img)
        print(label)
 
    # test_dataset = Data_Loader("test.txt", False)
    # print("数据个数：", len(test_dataset))
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=10,
    #                                            shuffle=True)
    # for image, label in test_loader:
    #     print(image.shape)
    #     print(label)
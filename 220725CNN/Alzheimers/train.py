'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-07-25 10:25:41
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-07-28 09:58:53
FilePath: \NTUAILab\220725CNN\Pneumonia\train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

'''
    Data Download: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
'''

train_root  = "E:\\NTUAILab\\Data\Alzheimer\\Alzheimer_s Dataset\\train"
test_root = "E:\\NTUAILab\\Data\\Alzheimer\\Alzheimer_s Dataset\\test"

#if wanted to display image 
from skimage import io
image = io.imread("E:\NTUAILab\Data\Alzheimer\Alzheimer_s Dataset\train\MildDemented\mildDem0.jpg")
print(image.shape)
io.imshow(image)

batch_size = 5

from keras.preprocessing.image import ImageDataGenerator

Generator = ImageDataGenerator()
train_data = Generator.flow_from_directory(train_root, (150, 150), batch_size=batch_size, shuffle= False)
test_data = Generator.flow_from_directory(test_root, (150, 150), batch_size=batch_size, shuffle= False)

#optional
print(train_data[0][0][0].shape)
# total 4317 data below to 5 clasess
print(len(train_data)) #4317/batch size
print(len(train_data[0])) #2, 1st image, 2nd is label
#print(train_data[0])
print(len(train_data[0][0])) #1st batch of 10 data
print(len(train_data[0][0][0])) #the image, the vertical
print(len(train_data[0][0][0][0])) #the image, the horizontal
print(len(train_data[0][0][0][0][0])) #the image, RGB

import tensorflow as tf
from matplotlib.pyplot import imshow
import os

im = train_data[0][0][0]
img = tf.keras.preprocessing.image.array_to_img(im)
imshow(img)

num_classes = len([i for i in os.listdir(train_root)])
print(num_classes)


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation="softmax"))
model.summary()


#remove optimizer if needed
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(train_data, batch_size = batch_size, epochs=2)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# score = model.evaluate(train_data)
# print(score)
score = model.evaluate(test_data)
print(score)


import seaborn as sns

pred = model.predict_classes(test_data)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_data.classes, pred)
sns.heatmap(cm, annot=True)


#depends on number of classes
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))


from keras.models import save_model
save_model(model, "Pneumonia")


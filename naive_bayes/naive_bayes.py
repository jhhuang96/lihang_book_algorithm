#encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 二值化 即将图像像素分为0或1
def binaryzation(img):
    cv_img = img.astype(np.uint8)   #修改数据类型为uint8：Unsigned integer (0 to 255)
    ret,cv_img = cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV)#图像中的灰度值大于50的重置像素值为0，否则为maxvalue=1
    return cv_img

def Train(trainset,train_labels):
    prior_probability = np.zeros(class_num)                         # 先验概率： 由于先验概率分母都是N，因此不用除于N，直接用分子即可。   因为对求极值没有影响
    conditional_probability = np.zeros((class_num,feature_len,2))   # 条件概率 这里class=10，feature_len = 784 ，2代表特征只有2个取值分别为0/1

    # 计算先验概率及条件概率
    for i in range(len(train_labels)):
        img = binaryzation(trainset[i])     # 图片二值化
        label = train_labels[i]

        prior_probability[label] += 1

        for j in range(feature_len):
            conditional_probability[label][j][img[j]] += 1

    # 将概率归到[1.10001]
    for i in range(class_num):
        for j in range(feature_len):

            # 经过二值化后图像只有0，1两种取值
            pix_0 = conditional_probability[i][j][0]    #像素取值为0的个数
            pix_1 = conditional_probability[i][j][1]    #像素取值为1的个数

            # 计算0，1像素点对应的条件概率
            probalility_0 = (float(pix_0)/float(pix_0+pix_1))*1000000 + 1  #像素取值为0的概率，并且映射到[1,1000000],防止出现概率值为0的情况,将小数扩大到整数
            probalility_1 = (float(pix_1)/float(pix_0+pix_1))*1000000 + 1  #像素取值为1的概率，并且映射到[1,1000000],防止出现概率值为0的情况,将小数扩大到整数

            conditional_probability[i][j][0] = probalility_0
            conditional_probability[i][j][1] = probalility_1

    return prior_probability,conditional_probability

# 计算概率
#由于Python 浮点数精度的原因，784个浮点数联乘后结果变为Inf，而Python中int可以无限相乘的，因此可以利用python int的特性对先验概率与条件概率进行一些改造。
def calculate_probability(img,label):
    probability = int(prior_probability[label])

    for i in range(len(img)):
        probability *= int(conditional_probability[label][i][img[i]])

    return probability

def Predict(testset,prior_probability,conditional_probability):
    predict = []

    for img in testset:

        # 图像二值化
        img = binaryzation(img)

        max_label = 0
        max_probability = calculate_probability(img,0)

        for j in range(1,10):
            probability = calculate_probability(img,j)

            if max_probability < probability:
                max_label = j
                max_probability = probability

        predict.append(max_label)

    return np.array(predict) #将列表化为矩阵，方便运算，列表无法运算


class_num = 10
feature_len = 784

if __name__ == '__main__':

    print ('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('D:/github/lihang_book_algorithm/data/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print ('read data cost ',time_2 - time_1,' second','\n')

    print ('Start training')
    prior_probability,conditional_probability = Train(train_features,train_labels)
    time_3 = time.time()
    print ('training cost ',time_3 - time_2,' second','\n')

    print ('Start predicting')
    test_predict = Predict(test_features,prior_probability,conditional_probability)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' second','\n')

    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)
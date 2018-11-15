#encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import random
import time
import operator

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 利用opencv获取图像hog特征
def get_hog_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('../hog.xml')

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features,(-1,324))

    return features

def Predict(inX,trainset,train_labels,k):
    dataSetSize = trainset.shape[0]
    diffMat= np.tile(inX,(dataSetSize,1)) - trainset
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #argsort函数返回的是数组值从小到大的索引值
    classCount={}
    for i in range(k):
        voteIlabel = train_labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) +1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse = True) 
    #reverse = True 降序:从大到小 ,itemgetter(1)代表根据第二个域进行排序
    return sortedClassCount[0][0] #返回发生频率最高的元素标签
    
if __name__ == '__main__':

    print ('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('D:/github/lihang_book_algorithm/data/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    features = get_hog_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print ('read data cost ',time_2 - time_1,' second','\n')

    print ('Start training')
    print ('knn do not need to train')
    time_3 = time.time()
    print ('training cost ',time_3 - time_2,' second','\n')

    print ('Start predicting')
    mTest = test_features.shape[0]
    test_predict = []
    for i in range(mTest):       
        test_predictI = Predict(test_features[i,:],train_features,train_labels,3)
        test_predict.append(test_predictI)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' second','\n')

    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)
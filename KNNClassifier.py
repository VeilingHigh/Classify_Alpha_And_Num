# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 09:50:15 2016

@author: VeilingHigh
"""
from PIL import Image
import numpy as np
import os
import csv
import operator
train_path_read="C:/Users/VeilingHigh/Desktop/project/trainResized/"
train_file_list=os.listdir(train_path_read)
train_file_list=sorted(train_file_list,key=lambda x:int(x.split('.')[0]))
test_path_read="C:/Users/VeilingHigh/Desktop/project/testResized/"
test_file_list=os.listdir(test_path_read)
test_file_list=sorted(test_file_list,key=lambda x:int(x.split('.')[0]))


train_csv_path="C:/Users/VeilingHigh/Desktop/project/trainLabels.csv"
test_csv_path="C:/Users/VeilingHigh/Desktop/project/testLabels.csv"
train_pic_items=[]
test_pic_items=[]
train_labels=[]
test_labels=[]
data=[]
num=0
totals=6220
def classify(inX,dataset,labels,k):
#knn算法函数
    diffMat=inX-dataset
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    #获取所有距离的值的索引，使其从小到大排列
    for i in range(k):
        votelabel=labels[sortedDistIndicies[i]]
        #通过索引，得到相关类别的标签
        classCount[votelabel]=classCount.get(votelabel,0)+1
        #给标签加字典，记录哪种标签出现次数最多
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
def write_labels(dat):
    csvfile_t=file(test_csv_path,'wb')
    writer=csv.writer(csvfile_t)
    writer.writerow(['ID','Class'])
    writer.writerows(dat)
    csvfile_t.close()
              
def get_labels(labels,csv_path):  
#获得类别标签函数
    csvfile=file(csv_path,'rb')
    reader=csv.reader(csvfile)
    for line in reader:
        labels.append(line[1])
    labels=labels[1:]
    csvfile.close()
    return labels
    
def loadImage(file_real):
#把图片变成矩阵
    im = Image.open(file_real)
    im = im.convert("L") 
    data = im.getdata()
    data = np.array(data)   
    return data

def get_pic_matrix(file_list,path_read,pic_items):
#把一堆图片变成矩阵
    for file_name in file_list:   
        file_real=path_read+file_name    
        data=loadImage(file_real)
        pic_items.append(data)
get_pic_matrix(train_file_list,train_path_read,train_pic_items)
#获得训练集的矩阵
train_labels=get_labels(train_labels,train_csv_path)
#获得训练集的标签
get_pic_matrix(test_file_list,test_path_read,test_pic_items)
#获得测试集的矩阵
def begin_work():
    global data
    for i in range(totals):
        predict_label=classify(test_pic_items[i],train_pic_items,train_labels,1)
        data.append((i+6284,predict_label))
        print 'num %d,the predict_label is %s'%(i+6284,predict_label)
        
begin_work()
write_labels(data)

       
        
        
    
    
    

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 09:50:15 2016

@author: VeilingHigh
"""
from PIL import Image
import numpy as np
import os
import csv
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=70)

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
    tr=np.array(pic_items)
    return tr
train_pic_items=get_pic_matrix(train_file_list,train_path_read,train_pic_items)
#获得训练集的矩阵
train_labels=get_labels(train_labels,train_csv_path)
#获得训练集的标签
test_pic_items=get_pic_matrix(test_file_list,test_path_read,test_pic_items)


clf = clf.fit(train_pic_items, train_labels)
aaa=clf.predict(test_pic_items)
aaa=list(aaa)
for i in range(totals):
    data.append((i+6284,aaa[i]))

write_labels(data) 
       
        
        
    
    
    


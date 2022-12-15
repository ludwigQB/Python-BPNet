import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utils.func import normalize
from Utils.func import sigmoid
##数据导入
data_train=pd.read_csv('PTA_T17-1.txt', sep=' ', header=None)
data_test=pd.read_csv('PTA_G17-1.txt', sep=' ', header=None)

(row, col)=data_train.shape
col=col-1                #去除读入的最后一列无效数据
(row_test, col_test)=data_test.shape
col_test=col_test-1

##定义网络参数
InputN=col-1  ##输入层节点数
HideN=50      ##隐藏层节点数
OutputN=1     ##输出层节点数

E_max=1e-4        ##截止误差
Train_num=5000    ##迭代次数
learnrate=1       ##学习速率
aerf=0.5          ##动量因子

##数据处理
train_max=np.max(data_train, axis=0)
train_min=np.min(data_train, axis=0)
data_train=np.array(data_train)
In_train=normalize(data_train, train_max, train_min, 0)

test_max=np.max(data_test, axis=0)
test_min=np.min(data_test, axis=0)
data_test=np.array(data_test)
In_test=normalize(data_test, test_max, test_min, 0)

x_train= In_train[:, 0:InputN] ##选取前17列为输入层
d_train= In_train[:, InputN:col]
labels_train= data_train[:, InputN:col]

x_test= In_test[:, 0:InputN]
labels_test= data_test[:, InputN:InputN + 1]

##网络权值初始化
Weight_in= 2*np.random.randn(InputN, HideN)
Weight_out= 2*np.random.randn(HideN, OutputN)
Delta_WIn=np.zeros((InputN, HideN))
Delta_WOut=np.zeros((HideN, OutputN))

##误差反向传播训练权值
E=np.zeros(Train_num)
Delta_Ho=np.ones((HideN, row))
Delta_Hin=np.ones((row, HideN))
Delta_Hout=np.ones((InputN, HideN, row))
fig,ax=plt.subplots()
y1=[]
for i in range(Train_num):
    Hin=np.dot(x_train, Weight_in)
    Hout=sigmoid(Hin)
    Opin=np.dot(Hout, Weight_out)
    Opot=sigmoid(Opin)

    E_p=(d_train - Opot)
    E_train= 0.5 * np.linalg.det(np.dot(E_p.T,E_p)) / row
    E[i]=E_train
    if E_train<E_max:
        print("提前完成训练目标")
        break
    Delta_ho= Opot * (1 - Opot) * E_p
    for j in range(row):
        Delta_Ho[:, j]= learnrate * Delta_ho[j] * Hout[j, :].T
    Delta_WOut= np.array([np.sum(Delta_Ho, axis=1)]).T / row + aerf * Delta_WOut

    for j in range(row):
        Delta_Hin[j, :]= Hout[j, :] * (1 - Hout[j, :]) * Weight_out.T * Delta_ho[j]

    for j in range(row):
        for k in range(HideN):
            Delta_Hout[:, k, j]= learnrate * Delta_Hin[j, k] * x_train[j, :].T
    Delta_WIn= np.sum(Delta_Hout, axis=2) / row + aerf * Delta_WIn

    Weight_in= Weight_in + Delta_WIn
    Weight_out= Weight_out + Delta_WOut
    if i%(Train_num/100)==0:
        y1.append(E_train)
        plt.ion()
        plt.title('Loss in training')
        plt.xlim(0,Train_num)
        plt.ylim(0,0.2)
        plt.scatter(i,E_train,color='r',linewidths=0.1)
        plt.pause(0.01)
plt.ioff()
plt.show()

Hin_train=np.dot(x_train,Weight_in)
Hout_train=sigmoid(Hin_train)
Opin_train=np.dot(Hout_train,Weight_out)
Opout_train=sigmoid(Opin_train)
pred_train=normalize(Opout_train, train_max[col - 1], train_min[col - 1], 1)
E1_train=np.abs(labels_train-pred_train)/labels_train
plt.figure()
plt.title('Relative Error in Training')
plt.plot(E1_train)
plt.show()

plt.figure()
plt.title('difference between labels and predictions in train')
plt.plot(labels_train,'r--',pred_train,'g--')
plt.legend(["labels","prediction"], loc='upper left')
plt.show()


Hin_test=np.dot(x_test,Weight_in)
Hout_test=sigmoid(Hin_test)
Opin_test=np.dot(Hout_test,Weight_out)
Opout_test=sigmoid(Opin_test)
pred_test=normalize(Opout_test, test_max[col - 1], test_min[col - 1], 1)
E_test=np.abs(labels_test-pred_test)/labels_test
plt.figure()
plt.title('Relative Error in Test')
plt.plot(E_test)
plt.show()

# E_mean=np.mean(E_test)
# E_std=np.std(E_test)
# E_rmse=np.sqrt(np.sum((labels_test-pred_test)*(labels_test-pred_test),axis=0)/row_test)
plt.figure()
plt.title('difference between labels and predictions in test')
plt.plot(labels_test,'r--',pred_test,'g--')
plt.legend(["labels","prediction"], loc='upper left')
plt.show()

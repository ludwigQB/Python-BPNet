import numpy as np
def normalize(data, max, min, type):
    m,n=data.shape
    output=np.ones((m,n))
    if type==0:
        for i in range(m):
            output[i,:]=(data[i,:]-min)/(max-min)
    if type==1:
        min_matrix=min*np.ones((m,n))
        for i in range(m):
            output[i,:]=(max-min)*data[i,:]+min_matrix[i,:]
    return output


def sigmoid(input):
    # (row,col)=input.shape
    # for i in range(row):
    #     for j in range(col):
    #         if input[i,j]<=0:
    #             input[i,j]=0
    # return input
    return 1/(1+np.exp(-input))
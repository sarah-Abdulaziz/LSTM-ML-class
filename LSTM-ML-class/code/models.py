# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:41:01 2018

@author: sara
"""
import pandas as pd
import numpy as np
import glob

from keras.models import Sequential
from keras.layers import Dense , LSTM, Activation
from sklearn.metrics import mean_squared_error
from statsmodels import api as sm
from math import sqrt
from sklearn.model_selection import train_test_split
np.random.seed(1)
total_spaces = {"NORREPORT": 65,
                   "SKOLEBAKKEN":512,
                   "SCANDCENTER": 1240,
                   "BRUUNS": 953,
                   "BUSGADEHUSET": 130,
                   "MAGASIN": 400,
                   "KALKVAERKSVEJ": 210,
                   "SALLING": 700,            
                   }



def split_data(df):
    df=df.values
    X=df[:,1]
    return X

def normalizeing(df):
    max_num=df.max()   
    normalize_list=[]
    for i in df:
          s= i / max_num
          normalize_list.append(s)
    norm_df=np.asarray(normalize_list)
    return norm_df

def LSTM_model(files_name,Xtrain, Xtest,Ytrain, Ytest):
    train_result=[]
    test_result=[]
    model = Sequential()
    model.add(LSTM(4, input_shape=(1,1)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adam',
                     metrics=['accuracy'])
    model.fit(Xtrain,Ytrain, epochs=1, batch_size=10)
    trainPredict = model.predict(Xtrain)      
    testPredict = model.predict(Xtest)
    for i in trainPredict[:,0] :
        train_result.append(i)
    for j in testPredict[:,0]:
        test_result.append(j)
    train_RMSE=RMSE(train_result,Ytrain)
    test_RMSE=RMSE(test_result,Ytest)
    return train_RMSE,test_RMSE

    
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
               a = dataset[i:(i+look_back)]
               dataX.append(a)
               dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def MA(train,n):
    acf = sm.tsa.acf(train,nlags=n-1)
    ma_y_hat=[]
    for i in  range(0,len(train)):
            ma_result=train[i:n]
            n=n+1
            ma_mu=np.sum(np.dot(ma_result,acf))/n
            alpha =np.random.randint(low=-1, high=1, size=1)
            MA = ma_mu + alpha
            for i in MA :
                ma_y_hat.append(i)
            if i == len(train) or n > len(train) :
                   break

    return ma_y_hat ,alpha ,acf


     
def Autoregressive(train, n) :
    acf = sm.tsa.acf(train,nlags=n-1)
    ma_y_hat=[]
    for i in  range(0,len(train)):
            ma_result=train[i:n]
            n=n+1
            ma_mu=np.sum(np.dot(ma_result,acf))/n
            alpha =np.random.randint(low=-1, high=1, size=1)
            MA = ma_mu + alpha
            for i in MA :
                ma_y_hat.append(i)
            if i == len(train) or n > len(train) :
                   break
#    result=RMSE(ma_y_hat,Ytrain[n:])
    return ma_y_hat

def gradient_descent(result,test):
    m=len(result)
    difference = test-result
    J=1.0/m * np.sum(difference)
    return J

def update_parameters(train,n,J,acf,alpha,learning_reat=0.01):
    update_alpha= alpha - learning_reat * J
    ma_y_hat=[]
    for i in  range(0,len(train)):
            ma_result=train[i:n]
            n=n+1
            ma_mu=np.sum(np.dot(ma_result,acf))/n
            NEW_MA = ma_mu + update_alpha
            for i in NEW_MA :
                ma_y_hat.append(i)
            if i == len(train) or n > len(train) :
                   break
    return ma_y_hat

def RMSE(result,test):
    rmse=sqrt(mean_squared_error(result, test))
    return rmse

def AR_model(files_name,Xtrain, Xtest,Ytrain, Ytest):
    result ,alpha ,acf=Autoregressive(Xtrain,7)
    rmse_before_update=RMSE(result,Ytrain[:len(result)])
    cost=gradient_descent(result,Ytrain[:len(result)] )
    NEW_MA=update_parameters(Xtrain,7,cost,acf,alpha)
    rmse_after_update=RMSE(NEW_MA,Ytrain[:len(result)])
    return NEW_MA ,rmse_after_update ,rmse_before_update

    
def AR_reshape(Xtrain, Xtest):
    Xtrain=np.reshape(Xtrain, (Xtrain.shape[0]))
    Xtest=np.reshape(Xtest, (Xtest.shape[0]))
    return  Xtrain, Xtest
    
def LSTM_reshape(Xtrain, Xtest):
    Xtrain=np.reshape(Xtrain, (Xtrain.shape[0],1,Xtrain.shape[1]))
    Xtest=np.reshape(Xtest, (Xtest.shape[0],1,Xtest.shape[1]))
    return  Xtrain, Xtest
    
if __name__ =='__main__':
    p="/*.csv"
    path=r'C:\Users\sara\Desktop\ML-class\hourly'
    allFiles = glob.glob(path + p)
    for files in allFiles:
          files_name = files.split("\\")[-1]
          files_name = files_name.split(".")[0]
          df=pd.read_csv(files,header=None)
          X=split_data(df)
          X=normalizeing(X)
          X,Y=create_dataset(X)
          Xtrain, Xtest,Ytrain, Ytest=train_test_split(X,Y, test_size=0.20)
          
#          ######################   new  AR MODEL ############################
#          AR_Xtrain, AR_Xtest=AR_reshape(Xtrain, Xtest)
#          NEW_MA ,rmse_after_update ,rmse_before_update=AR_model(files_name,AR_Xtrain, AR_Xtest,Ytrain, Ytest)
#          print(files_name ,"before_update",rmse_before_update,"after_update",rmse_after_update)
           ######################     AR MODEL ############################
#          AR_Xtrain, AR_Xtest=AR_reshape(Xtrain, Xtest)
#          NEW_MA ,rmse_after_update ,rmse_before_update=AR_model(files_name,AR_Xtrain, AR_Xtest,Ytrain, Ytest)
#          print(files_name ,"before_update",rmse_before_update,"after_update",rmse_after_update)
#          ######################   Autoregressive  MODEL ############################
#          AR_result=Autoregressive(AR_Xtrain, 3)
#          AR_rmse=RMSE(AR_result,Ytrain[2:])
#          print(files_name,AR_rmse)
#          ######################   LSTM MODEL ############################
          LSTM_Xtrain, LSTM_Xtest=LSTM_reshape(Xtrain, Xtest)
#          LSTM_model(files_name,LSTM_Xtrain, LSTM_Xtest,Ytrain, Ytest)
          train_RMSE,test_RMSE=LSTM_model(files_name,LSTM_Xtrain, LSTM_Xtest,Ytrain, Ytest)
          print(files_name,"train rmse",train_RMSE,"test rmse",test_RMSE)

          



##          
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:15:46 2020

@author: Manting
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

#%% matplotlib
def show_train_history_acc(title, train, val):
    plt.figure(figsize=(6, 4))
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[val])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

def show_train_history_loss(title, train, val):
    plt.figure(figsize=(6, 4))
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[val])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()  

#%% 讀資料，分割符號為tab(\t)
train_df = pd.read_csv('train.csv' ,delimiter = "\t").drop([1615])
test_df = pd.read_csv('test.csv' ,delimiter = "\t")
label_df = pd.read_csv('sample_submission (1).csv' ,delimiter = "\t")
label_df[['id','label']] = label_df['id,label'].str.split(",",expand = True) 
X_train = (train_df.loc[:,['text']].to_numpy()).reshape(-1)
X_test = (test_df.loc[:,['text']].to_numpy()).reshape(-1)
y_train = (train_df.loc[:,['label']].to_numpy()).reshape(-1).astype(int)
y_test = (label_df.loc[:,['label']].to_numpy()).reshape(-1).astype(int)

#%% 建立Token
token = Tokenizer(num_words = 3800) # 建立一個字典
token.fit_on_texts(X_train)  
token.word_index # 依照每個字在訓練集出現的次數進行排序
# 將文字轉成數字list
X_train_seq = token.texts_to_sequences(X_train) 
X_test_seq = token.texts_to_sequences(X_test)
# 截長補短，不足380個字的在前面補0
X_train = sequence.pad_sequences(X_train_seq, maxlen = 380)
X_test = sequence.pad_sequences(X_test_seq, maxlen = 380)

#%% RNN
modelRNN = Sequential()  
modelRNN.add(Embedding(output_dim = 32, # 將數字list轉換為32維度的向量
                       input_dim = 3800,  # 輸入的維度是字典的長度
                       input_length = 380)) # 截長補短
modelRNN.add(Dropout(0.7)) # 隨機在神經網路中放棄70%的神經元，避免overfitting
modelRNN.add(SimpleRNN(units = 16)) # 建立16個神經元的hidden layer
modelRNN.add(Dense(units = 256, activation = 'relu')) # 建立256個神經元的hidden layer，relu為激活函數
modelRNN.add(Dropout(0.7))  
modelRNN.add(Dense(units = 1 ,activation = 'sigmoid')) # 建立一個神經元的output layer，sigmoid為激活函數
modelRNN.summary()

# Loss function使用Cross entropy
# adam最優化方法可以更快收斂
modelRNN.compile(loss = 'binary_crossentropy',
     optimizer = 'adam',
     metrics = ['accuracy']) 

train_history = modelRNN.fit(X_train,y_train, 
         epochs = 10, # 執行10次訓練週期
         batch_size = 100, # 每一批次訓練100筆資料
         verbose = 2, # 顯示訓練過程
         validation_split = 0.2) # 設定80%訓練資料、20%驗證資料

# 評估準確率
scores = modelRNN.evaluate(X_test, y_test,verbose = 1)
print(scores[1])
print('mean loss:', np.mean(train_history.history["loss"]))
print('mean acc:', np.mean(train_history.history["accuracy"]))
show_train_history_acc('RNN Train History', 'accuracy', 'val_accuracy') 
show_train_history_loss('RNN Train History', 'loss', 'val_loss')

#%% LSTM
modelLSTM = Sequential() #建立模型
modelLSTM.add(Embedding(output_dim = 32,
                       input_dim = 3800,  
                       input_length = 380)) 
modelLSTM.add(Dropout(0.7))        
modelLSTM.add(LSTM(units = 32)) 
modelLSTM.add(Dense(units = 256 ,activation = 'relu')) 
#modelLSTM.add(Dense(units = 32 ,activation = 'relu')) 
modelRNN.add(Dropout(0.7))  
modelLSTM.add(Dense(units = 1 ,activation = 'sigmoid'))
modelLSTM .summary()
modelLSTM.compile(loss = 'binary_crossentropy',
     optimizer = 'adam',
     metrics = ['accuracy']) 

train_history = modelLSTM.fit(X_train, y_train, 
         epochs = 10, 
         batch_size = 100,
         verbose = 2,
         validation_split = 0.2)

scores = modelLSTM .evaluate(X_test, y_test,verbose = 1)
print(scores[1])
print('mean loss:', np.mean(train_history.history["loss"]))
print('mean acc:', np.mean(train_history.history["accuracy"]))
show_train_history_acc('LSTM Train History', 'accuracy', 'val_accuracy') 
show_train_history_loss('LSTM Train History', 'loss', 'val_loss')





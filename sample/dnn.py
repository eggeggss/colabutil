!rm -rf colabutil
!git clone https://github.com/eggeggss/colabutil.git

from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy
import sys
from colabutil.tools import util 

traindata,testdata=mnist.load_data()
x_train=traindata[0]
y_train=traindata[1]
x_test=testdata[0]
y_test=testdata[1]
print(y_test.shape)

#resharp

plt.gcf().set_size_inches(2,2)#設定大小
plt.imshow(x_train[0],cmap='binary')#顯示圖型
plt.title=y_train[0]

util.plot_images_labels_prediction(x_train,y_train, [],idx=100)

x_train_reshape=x_train.reshape(60000,784).astype('float32')
x_test_reshape=x_test.reshape(10000,784).astype('float32')

x_train_normal=x_train_reshape/255
x_test_normal=x_test_reshape/255

#one hot encoding
y_train_onehot=np_utils.to_categorical(y_train)
y_test_onehot=np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense

#定義Model

from keras.layers import Dropout
model=Sequential()
h1=Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu')
model.add(h1)
#units 這層256
#input_dim 輸入784
#初始weight 符合常態分配
#激活函數 relu

#每次訓練期間會隨機dropout 50%神經元
model.add(Dropout(0.5))
#加入Dropout 刪掉廢物神經元


h2=Dense(units=1000,kernel_initializer='normal',activation='relu')
model.add(h2)

model.add(Dropout(0.5))

h2=Dense(units=10,kernel_initializer='normal',activation='softmax')
model.add(h2)
#output 10


#模型摘要
model.summary()
#h0 = 784X256+256=200960 
#每個神經元有784個weight,256個神經元,每個神經元有256個bias,
#每個神經元最後外面包一個激活函數
#h1 = 256X10+10=2570

#設定loss funtion - crossentropy
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

validate=0.2 #驗證資料20%
epochs=10 #訓練10次
batch_size=300 #每一個epochs 300筆
print('x shape:',x_train_normal.shape)
print('y_shape:',y_train_onehot.shape)
history=model.fit(
                  x=x_train_normal,
                  y=y_train_onehot,
                  validation_split=validate,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=2)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc','val_acc'],loc='upper left')
#acc與val_acc間的差距表示有些微overfitting


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','val_loss'],loc='upper left')


metrics=model.evaluate(x_test_normal,y_test_onehot)
print('test_loss=',metrics[0],'test_acc=',metrics[1])


#print(x_test_normal[1:].shape)
#predict 第3筆
number=0
print(x_test_normal[number:].shape)

predition=model.predict_classes(x_test_normal[number:])

print('predition shape:',predition.shape)
print('預測:',predition[0],'答案:',y_test[number])

plt.gcf().set_size_inches(2,2)#設定大小

plt.imshow(x_test[number],cmap='binary')#顯示圖型


preditions=model.predict_classes(x_test_normal)

util.plot_images_labels_prediction(x_test,y_test,preditions)


#Cunfusion matrix


import pandas as pd
print(preditions.shape,'/',y_test.shape)

pd.crosstab(preditions,y_test,rownames=['predict'],colnames=['real'])


print(y_test.shape,'/',preditions.shape)
df = pd.DataFrame( {'label':y_test,'predict':preditions })

df[(df.label==5)&(df.predict==3)]

plt.imshow(x_test[1393],cmap='binary')








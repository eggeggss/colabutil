# colabutil
colabutil

<pre>

Keras快速api

第一層dnn
Dense(units,input_dim,kernel_initializer='normal',activation='relu')

第一層cnn
cnn1=Conv2D(input_shape=(64,64,3),filters=filter_1,kernel_size=(3,3),padding='same',activation='relu')

MaxPooling2D(x,x)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,validation_split,epochs,batch_size,verbose)

metric=model.evaluate(x_test_normal,y_test_onehot)

meric[0] =>loss ,mertric[1] =>acc

</pre>

<pre>
from keras import optimizers 
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              #optimizer='adam',
              metrics=['accuracy'])
              
</pre>
<pre>
from colabutil.tools import util
#取得image路徑
image_train_list=prepare_image_dir('train')

#路徑圖片resize並轉成numpy array
x_train_image=image_list_to_nparray(image_train_list,height,width,channel)

#顯示圖片
util.plot_images_labels_prediction(x_train_image,y_train_label,[],0)

#畫grad_history
util.draw_history(grad_historys,'acc')

#解壓縮
util.gzip('gdata.zip')

</pre>


regulation 降低overfiting

爛打越小bias越大(overfitting)

爛打越大variance越小(underfitting)

複雜的模型越容易造成overfiting
資料不足
訓練過度
模型複雜

解決:
降低深度與寬度

regulization

dropout

early stoping

image data argument


梯度下降法
sgd

動量

momentum sgd加上動量,會錯過global minmum

nag下預見上坡減速,momentum改良

自適應

adagrad - learning rate 越來越小,缺點最後不動

adadelta - learning rate 參考最近最近的梯度 , 不會無限小

rmsprop - 只加最近的梯度,跟adadelta相似

自適應+動量

adam - momentum + rmsprop

nadam=nag+adam

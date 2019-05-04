import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras import optimizers 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

!rm -rf colabutil
!git clone https://github.com/eggeggss/colabutil.git

# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('gdrive', force_remount=True)

!cp 'gdrive/My Drive/Colab Notebooks/TenLong/ex_Cat_Dog/gdata.zip' .

from colabutil.tools import util

util.gzip('gdata.zip')

image_path='gdata'

height=64
width=64
channel=3
batch_size=50

train_generator=ImageDataGenerator(rescale=1./255)
valid_generator=ImageDataGenerator(rescale=1./255)
test_generator=ImageDataGenerator(rescale=1./255)


train_dir=os.path.join(image_path,'train')

train_iterator=train_generator.flow_from_directory(
    directory=train_dir,
    batch_size=batch_size,
    target_size=(height,width),
    shuffle=True,
    class_mode='binary')


for batch_data,batch_label in train_iterator:
     print(batch_data.shape)
     print(batch_label.shape)
     break


valid_dir=os.path.join(image_path,'validation')

valid_iterator=valid_generator.flow_from_directory(
    directory=valid_dir,
    batch_size=batch_size,
    target_size=(height,width),
    shuffle=True,
    class_mode='binary'
)

test_dir=os.path.join(image_path,'test')

test_iterator=test_generator.flow_from_directory(
   directory=test_dir,
   batch_size=batch_size,
   target_size=(height,width),
   shuffle=True,
   class_mode='binary'
)


model=Sequential()
cnn1=Conv2D(
          input_shape=(height,width,channel),
          filters=64,
          kernel_size=(3,3),
          padding='same',
          activation='relu'
     )

maxpooling1=MaxPooling2D(2,2)


cnn2=Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding='same',
        activation='relu'
)

maxpooling2=MaxPooling2D(2,2)

flatten=Flatten()

h1=Dense(units=1024,activation='relu')

output=Dense(units=1,activation='sigmoid')

model.add(cnn1)
model.add(maxpooling1)
model.add(cnn2)
model.add(maxpooling2)
model.add(flatten)
model.add(h1)
model.add(output)

model.summary()


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              #optimizer='adam',
              metrics=['accuracy'])

grad_history=model.fit_generator(
     generator=train_iterator,
     steps_per_epoch=train_iterator.samples//batch_size,
     epochs=10,
     validation_data=valid_iterator,
     validation_steps=valid_iterator.samples//batch_size,
     verbose=1
)

util.draw_history(grad_history,'loss')


util.draw_history(grad_history,'acc')

metric=model.evaluate_generator(test_iterator,test_iterator.samples//batch_size)
print('loss:',metric[0],'acc:',metric[1])


preditions=model.predict_generator(test_iterator,test_iterator.samples//batch_size)

preditions[:10]

for test_image,test_label in test_iterator:
     print(test_image.shape)
     print(test_label.shape)
     break

result=model.predict_classes(test_image)
      
util.plot_images_labels_prediction(test_image,test_label,result,0)







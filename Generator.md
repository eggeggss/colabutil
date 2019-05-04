

<pre>
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

</pre>

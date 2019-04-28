import matplotlib.pyplot as plt
import os
import zipfile
import random
from PIL import Image
import numpy as np

def prepare_image_dir(datatype):
    train_img_list=[]
    #遍歷gdata/train
    for root,dir,file in os.walk(datatype):
        for name in file:
            train_img_list.append(os.path.join(root,name))
      
    random.shuffle(train_img_list)
    return train_img_list

def image_list_to_nparray(image_list,height,width,channel):
  
    num=len(image_list)
    
    img_np_array=np.zeros((num,height,width,channel),dtype=np.uint8)
      
    for i,image_file_path in enumerate(image_list):
        img=Image.open(image_file_path)
        img=img.resize((height,width),Image.BILINEAR)
        img_np_array[i]=img

    return img_np_array

def gzip(gpath):
  local_zip = gpath
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall('.')
  zip_ref.close()

def draw_history(grad_historys,type):
  if 'acc' in type:
     plt.plot(grad_historys.history['acc'])
     plt.plot(grad_historys.history['val_acc'])
     plt.legend(['acc','val_acc'],loc='upper left')
  elif 'loss' in type:
     plt.plot(grad_historys.history['loss'])
     plt.plot(grad_historys.history['val_loss'])
     plt.legend(['loss','val_loss'],loc='upper left')


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def plot_images_labels_prediction(images,
                                  labels,
                                  prediction,
                                  idx=0):
    fig = plt.gcf().set_size_inches(12, 5)
    for i in range(0, 10):
        ax=plt.subplot(2,5,i+1)
        ax.imshow(images[idx], cmap='binary')
        ax.set_xticks([]);ax.set_yticks([])  
        title= "label=" +str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
        ax.set_title(title,fontsize=10) 
        idx+=1 
    plt.show()



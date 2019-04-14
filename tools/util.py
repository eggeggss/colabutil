import matplotlib.pyplot as plt

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



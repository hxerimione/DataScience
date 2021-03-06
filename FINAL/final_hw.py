######################################## NEW VERSION ##############################################




from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten,Dropout,BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
import sklearn.metrics as metrics
from keras.models import load_model

def train_test_model():
    root_path='/content/images_d'
train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

trainGen=train_datagen.flow_from_directory(os.path.join(root_path,'training_set'),
                                            target_size=(300,300),
                                            class_mode='categorical')

testGen=test_datagen.flow_from_directory(os.path.join(root_path,'test_set'),
                                           target_size=(300,300),
                                           class_mode='categorical')


model = Sequential([
        Input(shape=(300,300,3)),

        Conv2D(32,kernel_size=3,activation='relu'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),

        Conv2D(32,kernel_size=3,activation='relu'),
        MaxPooling2D(pool_size=2),

        Conv2D(64,kernel_size=3,activation='relu'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),

        Conv2D(64,kernel_size=3,activation='relu'),
        MaxPooling2D(pool_size=2),
        
        Dropout(0.5),
        Flatten(),
        Dense(64,activation='relu'),
        #Dropout(0.5),
        Dense(3,activation='softmax')
        ])

model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
epochs=7
batchs_size=32

history=model.fit(
    trainGen,
    epochs=epochs,
    steps_per_epoch=36000/batchs_size,
    #batch_size=batchs_size,
    validation_data=testGen,
    validation_steps=9000/batchs_size)

print(history.history)
print("train loss=", history.history['loss'][-1])
print("validation loss=", history.history['val_loss'][-1]) 

)


        

def show_graph(history_dict):
    accuracy = history_dict['acc']
    val_accuracy = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(16, 1))
    
    plt.subplot(121)
    plt.subplots_adjust(top=2)
    plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Trainging and validation accuracy and loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=5)

    plt.subplot(122)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

    plt.show()

def classification(index):
    test_datagen=ImageDataGenerator(rescale=1./255)

     testGen=test_datagen.flow_from_directory(os.path.join(root_path,'test_set'),
                                               target_size=(300,300),
                                               batch_size=500,
                                               class_mode='categorical')

    model = load_model('/content/drive/MyDrive/Colab Notebooks/model-201811291')
    X_test,y_test=next(testGen)
    trueindex=np.zeros(len(y_test))
    for i in range(len(y_test)):
        if(np.argwhere(y_test[i]==1)[0]==index): 
            trueindex[i]=1
  
    predindex=model.predict(X_test)
    predindex=np.argmax(predindex,axis=1)
    print(trueindex)

    for i in range(len(y_test)):
        if(predindex[i]==index): 
            predindex[i]=1
        else:
            predindex[i]=0
    tp,tn,fp,fn=0,0,0,0
    for i in range(len(y_test)):
        if(trueindex[i]==1 and predindex[i]==1):
            tp=tp+1
        elif(trueindex[i]==0 and predindex[i]==0):
            tn=tn+1
        elif(trueindex[i]==1 and predindex[i]==0):
            fp=fp+1
        else:
            fn=fn+1
    accuracy=metrics.accuracy_score(trueindex,predindex)
    precision=metrics.precision_score(trueindex,predindex) 
    recall=metrics.recall_score(trueindex,predindex) 
    f1=metrics.f1_score(trueindex,predindex) 
    
    return accuracy,precision,recall,f1,tp,tn,fp,fn




def predict_image_sample(model, X_test, y_test, test_id=-1):
    if test_id < 0:
        from random import randrange
        test_sample_id = randrange(32)
    else:
        test_sample_id = test_id
        
    test_image = X_test[test_sample_id]

    
    plt.imshow(test_image)
    test_image = test_image.reshape(1,300,300,3)

    y_actual = y_test[test_sample_id]
    y_actual=np.argwhere(y_actual==1)
    print("y_actual number=", y_actual)
    
    y_pred = model.predict(test_image)
    print("y_pred=", y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    if y_pred==0:
      print("?????? : ??????")
    elif y_pred==1:
      print("?????? : ??????")
    else:
      print("?????? : ??????")

    if y_actual[0]==0:
      print("?????? : ??????")
    elif y_actual[0]==1:
      print("?????? : ??????")
    else:
      print("?????? : ??????")
    
    label_index = ['??????','??????','??????']
    if y_pred != y_actual:
        print("wrong!" )
        with open("wrong_samples.txt", "a") as errfile:
            print("%d"%test_sample_id, file=errfile)
    else:
        print("correct!")











train_test_model
show_graph(history.history)
#food
accuracy,precision,recall,f1,tp,tn,fp,fn = classification(0)
print('food??? accuracy : ',accuracy)
print('food??? precision : ',precision)
print('food??? recall : ',recall)
print('food??? f1-score : ',f1)
print('food??? tp : ',tp)
print('food??? tn : ',tn)
print('food??? fp : ',fp)
print('food??? fn : ',fn)
#interior
accuracy,precision,recall,f1,tp,tn,fp,fn = classification(1)
print('interior??? accuracy : ',accuracy)
print('interior??? precision : ',precision)
print('interior??? recall : ',recall)
print('interior??? f1-score : ',f1)
print('interior??? tp : ',tp)
print('interior??? tn : ',tn)
print('interior??? fp : ',fp)
print('interior??? fn : ',fn)
#exterior
accuracy,precision,recall,f1,tp,tn,fp,fn = classification(2)
print('exterior??? accuracy : ',accuracy)
print('exterior??? precision : ',precision)
print('exterior??? recall : ',recall)
print('exterior??? f1-score : ',f1)
print('exterior??? tp : ',tp)
print('exterior??? tn : ',tn)
print('exterior??? fp : ',fp)
print('exterior??? fn : ',fn)


test_datagen=ImageDataGenerator(rescale=1./255)

testGen=test_datagen.flow_from_directory(os.path.join(root_path,'test_set'),
                                           target_size=(300,300),
                                           batch_size=500,
                                           class_mode='categorical')

model = load_model('/content/drive/MyDrive/Colab Notebooks/model-201811291')
X_test,y_test=next(testGen)
predict_image_sample(model,X_test,y_test)


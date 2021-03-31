import numpy as np
import cv2

from glob import glob
import cv2
import matplotlib.pyplot as plt
import os

am = np.load('audioData.npy')
print(np.shape(am))
print(am.shape[0])
print(am.shape[1])
bd = np.mean(am, axis=0)
bd1 = np.mean(am, axis=1)

print(np.shape(bd))
print(np.shape)


#media = np.mean(banda0)
#print(media)
#saida = (banda0 - media)/(np.std(banda0))
#print(np.shape(saida))
#saida = saida.reshape(banda0,(banda0.shape[0],407,1))
#print(np.shape(saida))



frames = np.load('imagedata.npy')
tam = frames.shape[0]
#print(tam)
frames = np.reshape(frames,(frames.shape[0],224,224,3))
#print(np.shape(frames))
tensor = frames

for i in range(0,tam):
    t_0 = tensor[:,:,:,0]
    t_1 = tensor[:,:,:,1]
    t_2 = tensor[:,:,:,2]

    media0 = np.mean(t_0)
    media1 = np.mean(t_1)
    media2 = np.mean(t_2)

    norm0= (t_0 - media0)/(np.std(t_0))
    norm1= (t_1 - media1)/(np.std(t_1))
    norm2= (t_2 - media2)/(np.std(t_2))

    tensor[:,:,:,0] = norm0
    tensor[:,:,:,1] = norm1
    tensor[:,:,:,2] = norm2

    
#print(np.shape(tensor))
dim1=tensor.shape[0]
#print(dim1)
dim2=tensor.shape[1]
#print(dim2)

#----------------------------------------------------------------------------------
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
import tensorflow 

convolutional_layer = VGG16(weights='imagenet', include_top=False, input_shape= tensor )

def modelo():
    model_layer = Sequential()

    for layer in convolutional_layer.layers[:]:
        layer.trainable = False     #Freezes all layers in the vgg16
        model_layer.add(layer)

    model_layer.add(tensorflow.keras.layers.GlobalAveragePooling2D())
    model_layer.add(tensorflow.keras.layers.Dense(128, activation='tanh'))
    model_layer.add(tensorflow.keras.layers.Dense(1, activation='linear'))

    model_layer.compile(  optimizer='adam',
                        loss='mse')

    
   
    return model_layer

model = modelo()
model.summary() 


#saida = np.asarray(am.shape[0],bd)


model.fit( tensor, bd1 ,epochs=5)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")

#modelo() = keras.models.model_from_json(open('architecture.json').read())
#modelo().load_weights('model_weights.h5')


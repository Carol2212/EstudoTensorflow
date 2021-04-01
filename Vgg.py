#----> Importando os pacotes
import numpy as np
import cv2
import tensorflow 
from glob import glob
import cv2
import matplotlib.pyplot as plt
import os


#----> Importando a Vgg16
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense


#---->  Carregando as amostras de audio do arquivo numpy
audio = np.load('audioData.npy')
amostras = np.mean(audio, axis=1) # Media da posicao 1 
amostras = np.reshape(amostras,(amostras.shape[0],1))


# ----> Carregando os frames do arquivo numpy 
frames = np.load('imagedata.npy')
frames = np.reshape(frames,(frames.shape[0],224,224,3))  


#----> Normalização dos frames por canal 
t_0 = frames[:,:,:,0]
t_1 = frames[:,:,:,1]
t_2 = frames[:,:,:,2]

media0 = np.mean(t_0)
media1 = np.mean(t_1)
media2 = np.mean(t_2)

norm0= (t_0 - media0)/(np.std(t_0))
norm1= (t_1 - media1)/(np.std(t_1))
norm2= (t_2 - media2)/(np.std(t_2))

frames[:,:,:,0] = norm0
frames[:,:,:,1] = norm1
frames[:,:,:,2] = norm2


#----> Definindo e retornando o modelo 
def modelo():
    
    # weights - carrego os pesos da VGG treinada
    # include_top - baixo apenas as camadas convolucionais
    # input_shape - formato da entrada ,ou seja , o formato da imagem
    convolutional_layer = VGG16(weights='imagenet', include_top=False, input_shape= (224,224,3))
    
    model_layer = Sequential()

    for layer in convolutional_layer.layers[:]:
        layer.trainable = False     #Congelando os pesos treinados da VGG16
        model_layer.add(layer)

    
    model_layer.add(tensorflow.keras.layers.GlobalAveragePooling2D())
    model_layer.add(tensorflow.keras.layers.Dense(128, activation='tanh'))
    model_layer.add(tensorflow.keras.layers.Dense(1, activation='linear'))

    model_layer.compile(  optimizer='adam',
                        loss='mse')

    return model_layer

#----> Resumo do modelo 
model = modelo()
model.summary() 


#----> Treinamento dos dados 
model.fit( frames, amostras ,epochs=5)


#----> Carregando os pesos no arquivo "model.json"
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


#----> Salvando os pesos
model.save_weights("model.h5")



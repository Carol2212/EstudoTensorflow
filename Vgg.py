#----> Importando os pacotes
import numpy as np
import cv2
import tensorflow 
from glob import glob
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

#----> Importando a Vgg16
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import tensorflow

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# ----> Conjunto de teste

#----> Carregando as amostras de audio do arquivo numpy
audio_teste = np.load('M2U00001MPG/audioData.npy')
amostras_teste = np.mean(audio_teste, axis=1) # Media da posicao 1 
amostras_teste = np.reshape(amostras_teste,(amostras_teste.shape[0],1))

# ----> Carregando os frames do arquivo numpy 
frames_teste = np.load('M2U00001MPG/imagedata.npy')
print(frames_teste.shape)
frames_teste = np.reshape(frames_teste,(frames_teste.shape[0],224,224,3))  

#----> Normalização dos frames por canal 
tensor_teste0 = frames_teste[:,:,:,0]
tensor_teste1 = frames_teste[:,:,:,1]
tensor_teste2 = frames_teste[:,:,:,2]

media_teste0 = np.mean(tensor_teste0)
media_teste1 = np.mean(tensor_teste1)
media_teste2 = np.mean(tensor_teste2)

norm_teste0= (tensor_teste0 - media_teste0)/(np.std(tensor_teste0))
norm_teste1= (tensor_teste1 - media_teste1)/(np.std(tensor_teste1))
norm_teste2= (tensor_teste2 - media_teste2)/(np.std(tensor_teste2))

frames_teste[:,:,:,0] = norm_teste0
frames_teste[:,:,:,1] = norm_teste1
frames_teste[:,:,:,2] = norm_teste2


#----> Conjunto de treino

#----> Carregando as amostras de audio do arquivo numpy
audio_treino = np.load('M2U00002MPG/audioData.npy')
amostras_treino = np.mean(audio_treino, axis=1) # Media da posicao 1 
amostras_treino = np.reshape(amostras_treino,(amostras_treino.shape[0],1))

# ----> Carregando os frames do arquivo numpy 
frames_treino = np.load('M2U00002MPG/imagedata.npy')
frames_treino = np.reshape(frames_treino,(frames_treino.shape[0],224,224,3))  

#----> Normalização dos frames por canal 
tensor_treino0 = frames_treino[:,:,:,0]
tensor_treino1 = frames_treino[:,:,:,1]
tensor_treino2 = frames_treino[:,:,:,2]

media0 = np.mean(tensor_treino0)
media1 = np.mean(tensor_treino1)
media2 = np.mean(tensor_treino2)

norm_treino0= (tensor_treino0 - media0)/(np.std(tensor_treino0))
norm_treino1= (tensor_treino1 - media1)/(np.std(tensor_treino1))
norm_treino2= (tensor_treino2 - media2)/(np.std(tensor_treino2))

frames_treino[:,:,:,0] = norm_treino0
frames_treino[:,:,:,1] = norm_treino1
frames_treino[:,:,:,2] = norm_treino2

#----> Definindo o modelo 
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
#model.summary() 

#----> Treinamento 
print("frames treino:",frames_treino.shape)
print("amostras treino",amostras_treino.shape)

print("frames teste:",frames_teste.shape)
print("amostras teste",amostras_teste.shape)
print(type(frames_teste))
print(type(amostras_teste))

#----> O método fit() um gera um arquivo com os valores de perda e valores métricos durante o treinamento
dados_treino = model.fit(frames_treino, amostras_treino, validation_data=(frames_teste, amostras_teste), epochs=5)

#----> Carregando os pesos no arquivo "model.json"
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#----> Salvando os pesos
model.save_weights("model.h5")

#----> Avaliacao da acuracia e da perda (comparo o modelo com o conjunto de dados de teste)
frames_teste=frames_teste.astype('int32')
amostras_teste=amostras_teste.astype('int32')
print(type(frames_teste))
print(type(amostras_teste))

evaluate = model.evaluate(frames_teste, amostras_teste, verbose=2)
print('\nAvaliacao:',evaluate)

#----> Fazendo a predicao 
predicao = model.predict(frames_teste)
print('Formato da predicao:',predicao.shape)
print('Predicao 0:',predicao[0])

#----> Graficos de treinamento
with open("dados_treino.csv",'w') as data: 
   dados_treino = pd.read_csv(data, index_col=0)
dados_treino.plot()
plt.show()






















































































































































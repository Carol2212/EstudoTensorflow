
#----> Importando os pacotes
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow.keras.models import Sequential

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

entrada_teste = np.load('input_testing_data_GAP_fold_5.npy')
entrada_treino = np.load('input_training_data_GAP_fold_5.npy')
print(entrada_teste.shape)
print(entrada_treino.shape)

saida_teste = np.load('output_testing_data_fold_5.npy')
saida_treino = np.load('output_training_data_fold_5.npy')

saida_teste = np.reshape(saida_teste,(saida_teste.shape[0],1))
saida_treino = np.reshape(saida_treino,(saida_treino.shape[0],1))
print(saida_teste.shape)
print(saida_treino.shape)

#----> Definindo o modelo 

#model = tensorflow.keras.Sequential([
    #keras.layers.Flatten(input_shape=(28, 28)),
#    tensorflow.keras.layers.Dense(128, activation='relu'), #verificat tangente hiperbolica depois!!!
#    tensorflow.keras.layers.Dense(1, activation='linear')
#])
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(512,)))	
#model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.summary()

model.compile(  optimizer='adam',
                        loss='mse')

# nao uso acuracia porque quero uma aproximaçao;a acuracia eh definida em relaçao a 
# quantos acertos e quantos erros.Isso funciona em problemas de 
# classificaçao.Aq ,estamos fazendo uma regressao e meu modelo quer aproximar a funçao.        


#----> Treinamento dos dados 

#----> O método fit() um gera um arquivo com os valores de perda e valores métricos durante o treinamento
dados_treino = model.fit(entrada_treino, saida_treino,validation_data=(entrada_teste, saida_teste), epochs=20)


#----> Carregando os pesos no arquivo "model.json"
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


#----> Salvando os pesos
model.save_weights("model.h5")


#----> Avaliacao da acuracia e da perda (comparo o modelo com o conjunto de dados de teste)
test_loss = model.evaluate(entrada_teste, saida_teste, verbose=2)
print('\n Loss do modelo de teste:\n',test_loss)


#----> Fazendo a predicao 
predicao = model.predict(entrada_teste)
print(predicao.shape)
print(predicao[0])


#----> Graficos de treinamento
dados_treino_df = pd.DataFrame(dados_treino.history)

with open('dados_treino.csv', mode='w') as f:
    dados_treino_df.to_csv(f)


with open("dados_treino.csv",'r') as data: 
   dados_treino = pd.read_csv(data, index_col=0)
dados_treino.plot()
plt.show()






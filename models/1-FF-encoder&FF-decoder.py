# Backend and Import
import os
import scipy as scipy
import random as rn
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import optimizers
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau

# Hyperparameters
np.random.seed(7788)  
rn.seed(728)
first_para = 128
second_para = 80
encoding_dim = 512  #
input_dim = original_dim
output_dim = original_dim
batch_size = 32
Epochs = 200
learning_rate = 0.001  
StoppingPatience = 10
logdir='log_out_path'

# Load Data
rnaseq_file = os.path.join('/data/backup2/lingli/Proj_1_AE_recovery/15_reprocess_20201105/gene_expression_processed/log2tpm+1_mad_gene_0_1_scaled_9568samples_10240genes.csv')
rnaseq_df = pd.read_csv(rnaseq_file, index_col=0)
print('whole rnaseq dim', rnaseq_df.shape)




original_dim = rnaseq_df.shape[1]
rnaseq_df_tolist = rnaseq_df.values.tolist()
train_num = round(rnaseq_df.shape[0]*0.9)

x_train_original = rn.sample(rnaseq_df_tolist, train_num)
x_train_original_toarr = np.array(x_train_original)
print('x_train original dim', x_train_original_toarr.shape)


x_train_original_index = []
for sample_i in x_train_original:
    x_train_original_index.append(rnaseq_df_tolist.index(sample_i))

len(x_train_original_index)
type(x_train_original_index) # list
np.savetxt("./output/v5x_train_original_row_number.csv", x_train_original_index, delimiter=",", fmt='%d')


whole_index = []
for i in range(rnaseq_df.shape[0]):
    whole_index.append(i)

len(whole_index)  #

x_test_original_index = list(set(whole_index) - set(x_train_original_index))
len(x_test_original_index)
np.savetxt("./output/v5x_test_original_row_number.csv", x_test_original_index, delimiter=",", fmt='%d')


x_test_original = [item for item in rnaseq_df_tolist if not item in x_train_original]
len(x_test_original)

x_test_original_toarr = np.array(x_test_original)
print('x test original dim', x_test_original_toarr.shape)



x_train = np.reshape(x_train_original_toarr, (len(x_train_original_toarr), first_para, second_para, 1))
x_test = np.reshape(x_test_original_toarr, (len(x_test_original_toarr), first_para, second_para, 1))

print(x_train.shape, x_test.shape)  


# Convolutional autoEncoder
input_img = Input(shape=(first_para, second_para, 1))
x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x) # 10*16*64

flat = Flatten()(encoded)
hidden = Dense(encoding_dim)(flat)
hidden_decode = Dense(16*10*32)(hidden)
decoder_reshape = Reshape((16, 10, 32))(hidden_decode)

y = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_reshape)
y = UpSampling2D((2, 2))(y)
y = Conv2D(64, (3, 3), activation='relu', padding='same')(y)
y = UpSampling2D((2, 2))(y)
y = Conv2D(128, (3, 3), activation='relu', padding='same')(y)
y = UpSampling2D((2, 2))(y)
decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(y)  # sigmoid softmax tanh

conv_autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=hidden)


encoded_input = Input(shape=(encoding_dim,))
deco = conv_autoencoder.layers[-9](encoded_input)
deco = conv_autoencoder.layers[-8](deco)
deco = conv_autoencoder.layers[-7](deco)
deco = conv_autoencoder.layers[-6](deco)
deco = conv_autoencoder.layers[-5](deco)
deco = conv_autoencoder.layers[-4](deco)
deco = conv_autoencoder.layers[-3](deco)
deco = conv_autoencoder.layers[-2](deco)
deco = conv_autoencoder.layers[-1](deco)

decoder = Model(input=encoded_input, output=deco)

parallel_conv_autoencoder_model = multi_gpu_model(conv_autoencoder, gpus=3)
parallel_conv_autoencoder_model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

# Train autoEncoder
tbCallBack = TensorBoard(log_dir=logdir, 
                 histogram_freq=0,  
                 write_graph=True,  
                 write_grads=True, 
                 write_images=True,
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None
                 
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=StoppingPatience,
                                              verbose=1,  
                                              mode='auto',
                                              baseline=None)
                                              
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
                              factor=0.1,          
                              patience=10,
                              mode='auto')
                              
parallel_conv_autoencoder_model.fit(x_train, x_train,
                epochs=Epochs,
                batch_size=batch_size,
                shuffle=True,  
                validation_data=(x_test, x_test),
                callbacks=[
                          tbCallBack,
                          earlystopping,
                          reduce_lr,
                ])

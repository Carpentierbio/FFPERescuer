# Backend and Import
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, BatchNormalization
from keras.layers import Concatenate  
from keras.models import Model
from keras import backend as K
import scipy as scipy
import random as rn
import pandas as pd
import numpy as np
from keras import optimizers
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.callbacks import ReduceLROnPlateau


# Hyperparameters
np.random.seed(1234)  # for reproducibility
rn.seed(678)
encoding_dim = 512  #
batch_size = 32
Epochs = 100
learning_rate = 0.001
StoppingPatience = 30
dropoutfactor = 0.2
noise_factor2 = 0
noise_factor1 = 1-noise_factor2
mean = 0
sd = 1
img_rows = 128
img_cols = 80
first_para = 32
second_para = 32
logdir='log_out_path'


# Load Data
rnaseq_file = os.path.join('data/log2tpm+1_mad_gene_0_1_scaled_9568samples_1024genes_net2_model5.csv')
rnaseq_1024_df = pd.read_csv(rnaseq_file, index_col=0)

type(rnaseq_1024_df)
original_dim = rnaseq_1024_df.shape[1]
original_dim  # 1024

rnaseq_1024_df_tolist = rnaseq_1024_df.values.tolist()

train_index_file = os.path.join(dir_prefix+'15_reprocess_20201105/output/v5x_train_original_row_number.csv')
x_train_original_index = pd.read_csv(train_index_file, header=None)

x_train_original = []
for i in range(len(x_train_original_index)):
        x_train_original.append(rnaseq_1024_df_tolist[ x_train_original_index.iloc[i,0]])

x_train_original_toarr = np.array(x_train_original)
print('x train original dim', x_train_original_toarr.shape)  # 8611*1024

test_index_file = os.path.join(dir_prefix+'15_reprocess_20201105/output/v5x_test_original_row_number.csv')
test_index_df = pd.read_csv(test_index_file,  header=None) 
x_test_original_index = test_index_df

x_test_original = []
for i in range(len(x_test_original_index)):
    x_test_original.append(rnaseq_1024_df_tolist[ x_test_original_index.iloc[i,0]])

x_test_original_toarr = np.array(x_test_original)
print('x test original dim', x_test_original_toarr.shape)

x_train = np.reshape(x_train_original_toarr, (len(x_train_original_toarr), 32, 32, 1))
x_test = np.reshape(x_test_original_toarr, (len(x_test_original_toarr), 32, 32, 1))

x_train_noisy = noise_factor1 * x_train + noise_factor2 * np.random.normal(loc=mean, scale=sd, size=x_train.shape)
x_test_noisy = noise_factor1 * x_test + noise_factor2 * np.random.normal(loc=mean, scale=sd, size=x_test.shape)


# Using the features from FF-encoder to train partial FF-encoder
rnaseq_file = os.path.join(dir_prefix+'15_reprocess_20201105/gene_expression_processed/log2tpm+1_mad_gene_0_1_scaled_9568samples_10240genes.csv')
rnaseq_flat_recovered_from_1024inputdf = pd.read_csv(rnaseq_file, index_col=0)
print('1st stage input dim', rnaseq_flat_recovered_from_1024inputdf.shape)
rnaseq_flat_recovered_from_1024inputdf_tolist = rnaseq_flat_recovered_from_1024inputdf.values.tolist()


x_train_for_pred = []
for i in range(len(x_train_original_index)):
    x_train_for_pred.append(rnaseq_flat_recovered_from_1024inputdf_tolist[x_train_original_index.iloc[i,0]])

x_train_for_pred = np.array(x_train_for_pred)
print('x train for predict', x_train_for_pred.shape) # 8611
x_train_predc = np.reshape(x_train_for_pred, (len(x_train_for_pred), img_rows, img_cols, 1))

x_test_for_pred = []
for i in range(len(x_test_original_index)):
    x_test_for_pred.append(rnaseq_flat_recovered_from_1024inputdf_tolist[x_test_original_index.iloc[i,0]])

x_test_for_pred = np.array(x_test_for_pred)
print('x test for predict', x_test_for_pred.shape) # 957
x_test_predc = np.reshape(x_test_for_pred, (len(x_test_for_pred), img_rows, img_cols, 1))

loaded_encoder = load_model("trained_models/v5encoder.h5")
pedicted_features_in_train = loaded_encoder.predict(x_train_predc)
pedicted_features_in_test = loaded_encoder.predict(x_test_predc)


# Partial FF-encoder
input_shape = (first_para, second_para, 1)
input_img = Input(input_shape)
x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
x = Dropout(dropoutfactor)(x)
flat = Flatten()(x)
out1 = Dense(encoding_dim)(flat)  

sec_net = Model(input=input_img, output=out1)
sec_net.summary()
sec_net.output_shape

parallel_sec_net_model = multi_gpu_model(sec_net, gpus=2)
parallel_sec_net_model.compile(optimizer=Adam(lr=learning_rate), loss='mse') 


# Train Partial FF-encoder
tbCallBack = TensorBoard(log_dir=logdir,  
                         histogram_freq=0,  
                         write_graph=True,  
                         write_grads=True,  
                         write_images=True, 
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
                              factor=0.5,
                              patience=20,
                              mode='auto')

earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=StoppingPatience,
                                              verbose=1,
                                              mode='auto',
                                              baseline=None)

parallel_sec_net_model.fit(x_train_noisy, pedicted_features_in_train,
                                    epochs=Epochs,
                                    batch_size=batch_size,
                                    verbose=1,
                                    shuffle=True,
                                    validation_data=(x_test_noisy, pedicted_features_in_test),
                                    callbacks=[tbCallBack])

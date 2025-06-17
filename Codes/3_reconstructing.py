
import os
os.chdir('/your/working/path/')

os.getcwd()
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, BatchNormalization
from keras.models import Model
from keras import backend as K
import scipy as scipy
import random as rn
import pandas as pd
import numpy as np
from keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

np.random.seed(1234)  # for reproducibility
rn.seed(678)

# Load Data to be reconstructed
rnaseq_file = os.path.join("./data/AMC_FFPE_CRC_log2tpm+1_feature_zero_scale_12samples_1024genes.csv")
rnaseq_1024_df = pd.read_csv(rnaseq_file, index_col=0) 


original_dim = rnaseq_1024_df.shape[1]

rnaseq_1024_df_tolist = rnaseq_1024_df.values.tolist()
rnaseq_1024_df_toarr = np.array(rnaseq_1024_df_tolist)

x_input = np.reshape(rnaseq_1024_df_toarr, (len(rnaseq_1024_df_toarr), 32, 32, 1))


## load partial FF-encoder and FF-decoder
loaded_network_no_noise = load_model("./models/Partial_FF_encoder.h5")
pedicted_features = loaded_network_no_noise.predict(x_input)

first_decoder = load_model("./models/FF_decoder.h5")
reconstructed = first_decoder.predict(pedicted_features)

flat_reconstructed = np.reshape(reconstructed, (len(reconstructed), 10240))
np.savetxt("./recovered_datasets/AMC_FFPE_CRC_reconstructed.csv", flat_reconstructed, delimiter=",")
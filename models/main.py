# 20201105
# https://blog.csdn.net/qq_23869697/article/details/85106365?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.compare&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.compare


# 设置tensorflow中使用tf.ConfigProto()，配置Session运行参数，GPU设备指定
# nvidia-smi

import os
os.chdir('/data/backup2/lingli/Proj_1_AE_recovery/15_reprocess_20201105')
os.getcwd()
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape
from keras.models import Model
from keras import backend as K
import scipy as scipy
import random as rn
import pandas as pd
import numpy as np
from keras import optimizers
from keras.optimizers import Adam


os.environ['CUDA_VISIBLE_DEVICES'] = '0,6,7'  # # 使用 GPU 2,3,4,5,6
## 动态申请显存
config = tf.ConfigProto()
config.gpu_options.allocator_type ='BFC'
## 当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存
config.gpu_options.allow_growth = True
## 当使用GPU时，设置GPU内存使用最大比例
config.gpu_options.per_process_gpu_memory_fraction = 0.9
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

import time
localtime = time.asctime(time.localtime(time.time()))
print("Start time :", localtime)

np.random.seed(7788)  # for reproducibility
rn.seed(728)

# Load Data
# 输入需要 样本在行 基因在列

# 6-input 没有基因筛选 没有Z score,没有[0,1] scale。
rnaseq_file = os.path.join('/data/backup2/lingli/Proj_1_AE_recovery/15_reprocess_20201105/gene_expression_processed/log2tpm+1_mad_gene_0_1_scaled_9568samples_10240genes.csv')
                                                                                                                    
rnaseq_df = pd.read_csv(rnaseq_file, index_col=0)
print('whole rnaseq dim', rnaseq_df.shape)

# pd.DataFrame(0, index=np.arange(len(rnaseq_file)), columns=feature_list)

type(rnaseq_df)
original_dim = rnaseq_df.shape[1]
original_dim  #

rnaseq_df_tolist = rnaseq_df.values.tolist()

train_num = round(rnaseq_df.shape[0]*0.9)
x_train_original = rn.sample(rnaseq_df_tolist, train_num)
x_train_original_toarr = np.array(x_train_original)
print('x_train original dim', x_train_original_toarr.shape)

# list.index只接受一个元素的输入
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

# x_test_original = rnaseq_df_tolist[x_test_original_index]

x_test_original = [item for item in rnaseq_df_tolist if not item in x_train_original]
len(x_test_original)

x_test_original_toarr = np.array(x_test_original)
print('x test original dim', x_test_original_toarr.shape)
# keras model expects the input to be in the form of [samples, time steps, features]

# input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
first_para = 128
second_para = 80

x_train = np.reshape(x_train_original_toarr, (len(x_train_original_toarr), first_para, second_para, 1))
x_test = np.reshape(x_test_original_toarr, (len(x_test_original_toarr), first_para, second_para, 1))

print(x_train.shape, x_test.shape)  

# set parameters
encoding_dim = 512  #
input_dim = original_dim
output_dim = original_dim
batch_size = 32
Epochs = 200
learning_rate = 0.001  # keras 默认0.001
print('Learning Rate:', learning_rate)

StoppingPatience = 10

# https://fooobar.com/questions/418926/keras-autoencoder-error-when-checking-target

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape
from keras.models import Model

# inputShape = (height, width, depth)
input_img = Input(shape=(first_para, second_para, 1))

# # v6 16 32 64;64 32 16
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
conv_autoencoder.summary()

# # this model maps an input to its encoded representation
encoder = Model(input=input_img, output=hidden)
encoder.summary()

# #### create the decoder model
# decoder = Model(input=latent_inputs, output=decoded)

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

# create the decoder model
decoder = Model(input=encoded_input, output=deco)
decoder.summary()

from keras.utils import multi_gpu_model

parallel_conv_autoencoder_model = multi_gpu_model(conv_autoencoder, gpus=3)

parallel_conv_autoencoder_model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
# parallel_conv_autoencoder_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
# parallel_conv_autoencoder_model.compile(optimizer=optimizers.SGD(lr=learning_rate), loss='mse')  # 把优化函数换了之后val loss正常下降了
# parallel_conv_autoencoder_model.compile(optimizer=optimizers.SGD(lr=learning_rate), loss='binary_crossentropy')  # 不行-2 -3
# parallel_conv_autoencoder_model.compile(optimizer='adadelta', loss='mse') # 不行

# 可视化
from keras.callbacks import TensorBoard

logdir='/data/backup2/lingli/Proj_1_AE_recovery/15_reprocess_20201105/log_out' # + datetime.now().strftime("%Y%m%d-%H%M%S")

tbCallBack = TensorBoard(log_dir=logdir,  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)

# tensorboard --logdir=/home/lingli/python_workspace/recovery/autoencoder/firebrowse_FF_gene_expression/7_conv_AE_input_10204_genes_9568_samples/log_1_12040 --port=6006

earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=StoppingPatience, # 能够容忍多少个epoch内都没有improvement
                                              verbose=1,  # 信息展示模式
                                              mode='auto',
                                              baseline=None,
                                              # restore_best_weights=False，
)

from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # 被监测的量
                              factor=0.1,          #  每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                              patience=10,
                              mode='auto')



parallel_conv_autoencoder_model.fit(x_train, x_train,
                epochs=Epochs,
                batch_size=batch_size,
                shuffle=True,  # 打乱样本
                validation_data=(x_test, x_test),
                callbacks=[
                          tbCallBack,
                          earlystopping,
                          reduce_lr,
                ])
                          
# save model
from keras.models import save_model
# import os

# 将Keras模型和权重保存在一个HDF5文件中
save_model(parallel_conv_autoencoder_model,'./models/v5_input_3-hidden_autoencoder.h5')   # 或者 autoencoder.save('sparse_3-hidden_autoencoder.h5')
save_model(encoder,'./models/v5encoder.h5')   # 或者 encoder.save('sparse_3-hidden_encoder.h5')
save_model(decoder,'./models/v5decoder.h5')

# # test
print('auencoder output: ', parallel_conv_autoencoder_model.predict(x_test[0:2]))
print('encoder + decoder output: ', decoder.predict(encoder.predict(x_test[0:2])))

# # 先删掉
# del parallel_conv_autoencoder_model  # deletes the existing model
#
# ## 加载保存的模型
from keras.models import load_model

# loaded_autoencoder = load_model("3-hidden_autoencoder.h5")
# print('test after load: ', loaded_autoencoder.predict(x_test[0:2]))  #
#
loaded_encoder = load_model("./models/v5encoder.h5")  # 这2个地方，如果前面把模型删除再load会报错，但是如果不删除encoder,decoder，后面依然可以预测。
# # loaded_decoder = load_model("3-hidden_decoder.h5")

# 利用encoder model提取特征并保存
rnaseq_df_tolist = rnaseq_df.values.tolist()
rnaseq_df_array = np.array(rnaseq_df_tolist)
rnaseq_df_reshape = np.reshape(rnaseq_df_array, (len(rnaseq_df_array), first_para, second_para, 1))

# encoded_features = encoder.predict(rnaseq_df_reshape)

encoded_features = loaded_encoder.predict(rnaseq_df_reshape)
type(encoded_features) # array
encoded_features_df = pd.DataFrame(encoded_features)
# 添加样本名称为行名
encoded_features_df.index = rnaseq_df._stat_axis.values.tolist()
encoded_features_df.head(5)

print('dim of encoded: ', encoded_features_df.shape)  # 在一个代码里，就直接用encoder,不用loaded_encoder了
np.savetxt("./output/v5_9568_512_3-hidden_encoded_features.1.csv", encoded_features_df, delimiter=",")

# 现在是只看test集里面的features


import time

localtime = time.asctime(time.localtime(time.time()))
print("End time :", localtime)
print('Learning Rate:', learning_rate)

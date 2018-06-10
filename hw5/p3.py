from keras.applications import InceptionV3
from reader import readShortVideo,getVideoList
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.applications.resnet50 import ResNet50
from keras.regularizers import *
import os
from skimage import io
import matplotlib.pyplot as plt
import time

#'''
train_path = '/root/data/DL_5/HW5_data/FullLengthVideos/videos/train'
train_label_path = '/root/data/DL_5/HW5_data/FullLengthVideos/labels/train/'
train_folder = sorted(os.listdir(train_path))

valid_path = '/root/data/DL_5/HW5_data/FullLengthVideos/videos/valid/'
valid_label_path = '/root/data/DL_5/HW5_data/FullLengthVideos/labels/valid/'
valid_folder = sorted(os.listdir(valid_path))


def get_data_label(path,label_path,folder_path,mode='train'):
    if(mode=='train'):
        num = 2994
    else:
        num = 2140
    all_X = np.zeros((len(folder_path),num,240,320,3),dtype=np.uint8)
    all_Y = np.zeros((len(folder_path),num,11))
    datalen = []
    for i,folder in enumerate(folder_path):
        print(folder)
        labeltxt_path = os.path.join(label_path,folder+'.txt')
        label = np.genfromtxt(labeltxt_path,dtype = 'int')
        train_path = os.path.join(path,folder)
        data = sorted(os.listdir(train_path))
        frames = []
        for j in range(len(data)):
            frame = io.imread(os.path.join(train_path,data[j]))
            frames.append(frame)
        all_X[i][:len(data)] = np.array(frames).astype(np.uint8)
        for k,n in enumerate(label):
            all_Y[i][k][n] = 1
        datalen.append(len(data))
    return all_X.reshape((-1,240,320,3)),all_Y,datalen

def get_frame(X,Y,data):
    data_X = np.zeros((len(X),500,2048),dtype = np.uint8)
    data_Y = np.zeros((len(Y),500,11))
    for i in range(len(X)):
        start_index = np.random.randint(0,data[i]-500,size=1)
        data_X[i] = X[i][start_index[0]:start_index[0]+500]
        data_Y[i] = Y[i][start_index[0]:start_index[0]+500]
    return data_X,data_Y






all_x,all_y,all_data = get_data_label(train_path,train_label_path,train_folder,mode='train')
valid_X,valid_Y,_ = get_data_label(valid_path,valid_label_path,valid_folder,mode = 'valid')


#train_X,train_Y  = get_frame(all_x,all_y,all_data)

input_tensor = Input(shape=(240,320,3))
model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False)
outcome = model.predict(all_x,verbose=1)
outcome_V = model.predict(valid_X,verbose = 1)

#train_X,trainMY  = get_frame(outcome,all_y,all_data)

#print(train_X.shape)
#print(train_Y.shape)
#print(valid_X.shape)
#print(valid_Y.shape)
#outcome = model.predict(train_X,verbose = 1)
#outcome_V = model.predict(valid_X.reshape((-1,240,320,3)),verbose =1)
#print(outcome_V)

#'''

input_features = Input(shape=(None,2048))
#x = Bidirectional(GRU(512,return_sequences=True))(iput_features)
state_h = Bidirectional(GRU(512,kernel_regularizer = l2(0.002),return_sequences = True,dropout = 0.2,recurrent_dropout = 0.2,stateful=False))(input_features)
state_h = Bidirectional(GRU(512,kernel_regularizer = l2(0.002),return_sequences = True,dropout = 0.2,recurrent_dropout = 0.2,stateful=False))(state_h)
state_h = Bidirectional(GRU(512,kernel_regularizer = l2(0.002),return_sequences = True,dropout = 0.2,recurrent_dropout = 0.2,stateful=False))(state_h)
state_h = Dense(11,activation='softmax')(state_h)
#state_h = Reshape((500,))(state_h)

videoRNN = Model(input_features,state_h)
videoRNN.summary()


videoRNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

checkpoint = ModelCheckpoint("./model/videos_RNN_model2_best.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max',period=1)

train_X,train_Y  = get_frame(outcome.reshape((-1,2994,2048)),all_y,all_data)
hist = videoRNN.fit(train_X.reshape((-1,500,2048)), train_Y, epochs=100, batch_size=256,validation_data=(outcome_V.reshape((-1,2140,2048)),valid_Y),callbacks=[checkpoint],verbose=1)
#see = videoRNN.predict(feature_V)
#np.savetxt('./see.npy',np.round(see),delimiter=',')
videoRNN.load_weights('./model/videos_RNN_model2_best.hdf5')
np.save('see.npy',np.around(videoRNN.predict(outcome_V.reshape((-1,2140,2048)))))
np.save('./p3_acc.npy',hist.history['val_acc'])
np.save('./p3_loss.npy',hist.history['loss'])

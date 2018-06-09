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
    all_X = np.zeros((len(folder_path),num,240,320,3))
    all_Y = np.zeros((len(folder_path),num))
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
        all_Y[i][:len(data)]
        datalen.append(len(data))
    return all_X,all_Y,train_data

def get_frame(X,Y,data):
    data_X = np.zeros((len(X),500,240,320,3))
    data_Y = np.zeros((len(Y),500))
    for i in range(len(X)):
        start_index = np.random.randint(0,data[i]-500,size=1)
        data_X[i] = X[i][start_index[0]:start_index[0]+500]
        data_Y[i] = Y[i][start_index[0]:start_index[0]+500]
    return data_X,data_Y






all_x,all_y,all_data = get_data_label(train_path,train_label_path,train_folder,mode='train')
valid_X,valid_Y,_ = get_data_label(valid_path,valid_label_path,valid_folder)
train_X,train_Y  = get_frame(all_x,all_y,all_data)

input_tensor = Input(shape=(240,320,3))
model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False)

outcome = model.predict(train_X,verbose = 1)
outcome_V = model.predict(valid_X,verbose =1)



input_features = Input(batch_shape=(None,None,2048))
#x = Bidirectional(GRU(512,return_sequences=True))(iput_features)
output,state_h = Bidirectional(GRU(256,return_state=True))(input_features)
state_h = Dense(11,activation='softmax')(state_h)
state_h = Reshape((11,))(state_h)

videoRNN = Model(input_feature,state_h)



videoRNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

checkpoint = ModelCheckpoint("./model/videos_RNN_model_{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max',period=1)

videoRNN.fit(outcome, train_Y, epochs=50, batch_size=64,validation_data=(outcome_V,valid_Y),callbacks=[checkpoint],verbose=1)
#see = videoRNN.predict(feature_V)
#np.savetxt('./see.npy',np.round(see),delimiter=',')



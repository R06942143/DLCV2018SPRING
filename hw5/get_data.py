from keras.applications import InceptionV3
from reader import readShortVideo,getVideoList
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import matplotlib.pyplot as plt


a = getVideoList('/root/data/DL_5/HW5_data/TrimmedVideos/label/gt_train.csv')
bb = []
train_X = []
valid_X = []
for i in range(len(list(a.values())[4])):
    b = readShortVideo('/root/data/DL_5/HW5_data/TrimmedVideos/video/train',
                 list(a.values())[4][i],list(a.values())[6][i])

    train_X.extend(list(np.array(b.tolist())))
    bb.append(len(b))
QQ = np.array(train_X).astype(np.uint8)
bb = np.array(bb)
ans = np.zeros((len(list(a.values())[0]),11))
for i in range(len(ans)):
    ans[i,int(list(a.values())[0][i])] = 1
np.save('/root/data/DL_5/train_Y.npy',ans)
np.save('/root/data/DL_5/train_X.npy',QQ)
np.save('/root/data/DL_5/train_bb.npy',bb)

a = getVideoList('/root/data/DL_5/HW5_data/TrimmedVideos/label/gt_valid.csv')
bb = []
train_X = []
valid_X = []
for i in range(len(list(a.values())[4])):
    # b = readShortVideo(r'H:\master\code\python\DLCV_HW5\data\HW5_data\TrimmedVideos\video\train',
    #              list(a.values())[4][i],list(a.values())[6][i])

    b = readShortVideo('/root/data/DL_5/HW5_data/TrimmedVideos/video/valid',
                 list(a.values())[4][i],list(a.values())[6][i])
    valid_X.extend(list(np.array(b.tolist())))
    bb.append(len(b))
QQ = np.array(valid_X).astype(np.uint8)
bb = np.array(bb)
ans = np.zeros((len(list(a.values())[0]),11))
for i in range(len(ans)):
    ans[i,int(list(a.values())[0][i])] = 1
np.save('/root/data/DL_5/valid_Y.npy',ans)
np.save('/root/data/DL_5/valid_X.npy',QQ)
np.save('/root/data/DL_5/valid_bb.npy',bb)



'''
a = getVideoList(r'H:\master\code\python\DLCV_HW5\data\HW5_data\TrimmedVideos\label\gt_train.csv')

train_X = np.load(r'H:\master\code\python\DLCV_HW5\data\train_X.npy')
train_Y = np.load(r'H:\master\code\python\DLCV_HW5\data\train_Y.npy')
valid_X = np.load(r'H:\master\code\python\DLCV_HW5\data\valid_X.npy')
valid_Y = np.load(r'H:\master\code\python\DLCV_HW5\data\valid_Y.npy')
# ans = np.zeros((3236,11))
# for i in range(len(ans)):
#     ans[i,int(list(a.values())[0][i])] = 1 
# np.save(r'H:\master\code\python\DLCV_HW5\data\train_Y.npy',ans)
bb      = np.load(r'H:\master\code\python\DLCV_HW5\data\bb.npy')
valid_bb= np.load(r'H:\master\code\python\DLCV_HW5\data\valid_bb.npy')
model = InceptionV3(include_top=True, weights='imagenet')

outcome = model.predict(train_X)
outcome_V = model.predict(valid_X)
feature_average = []
count = 0
for i in bb:
    for j in range(i):
        if(j==0):
            f_a = outcome[j+count]
        else:
            f_a +=outcome[j+count]
    f_a /=i
    count+=i
    feature_average.append(f_a)

feature_X = np.array(feature_average)

feature_average = []
count = 0
for i in valid_bb:
    for j in range(i):
        if(j==0):
            f_a = outcome_V[j+count]
        else:
            f_a +=outcome_V[j+count]
    f_a /=i
    count+=i
    feature_average.append(f_a)

feature_V = np.array(feature_average)


FC_layer = Sequential()
FC_layer.add(Dense(2048,input_shape = outcome[0].shape,activation = 'relu'))
FC_layer.add(Dropout(0.2))
FC_layer.add(BatchNormalization())
FC_layer.add(Dense(1024,input_shape = outcome[0].shape,activation = 'relu'))
FC_layer.add(Dropout(0.2))
FC_layer.add(BatchNormalization())
FC_layer.add(Dense(512,input_shape = outcome[0].shape,activation = 'relu'))
FC_layer.add(Dropout(0.2))
FC_layer.add(BatchNormalization())
FC_layer.add(Dense(256,input_shape = outcome[0].shape,activation = 'relu'))
FC_layer.add(Dropout(0.2))
FC_layer.add(BatchNormalization())
FC_layer.add(Dense(11,input_shape = outcome[0].shape,activation = 'softmax'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

checkpoint = ModelCheckpoint(r'H:\master\code\python\DLCV_HW5\model'+"\model_{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max',period=1)

model.fit(feature_X, train_Y, epochs=50, batch_size=64,validation_data=(feature_V,valid_Y),callbacks=[checkpoint],verbose=1)
'''




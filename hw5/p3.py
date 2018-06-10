from reader import readShortVideo,getVideoList
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.applications.resnet50 import ResNet50
from keras.regularizers import *
import os
from skimage import io
import sys
valid_path = sys.argv[1]
valid_folder = sorted(os.listdir(valid_path))


def get_data_label(path,folder):
    print(folder)
    train_path = os.path.join(path,folder)
    data = sorted(os.listdir(train_path))
    frames = []
    for j in range(len(data)):
        frame = io.imread(os.path.join(train_path,data[j]))
        
        frames.append(frame)
    return np.array(frames).astype(np.uint8)




valid_X1 = get_data_label(valid_path,valid_folder[0])
valid_X2 = get_data_label(valid_path,valid_folder[1])
valid_X3 = get_data_label(valid_path,valid_folder[2])
valid_X4 = get_data_label(valid_path,valid_folder[3])
valid_X5 = get_data_label(valid_path,valid_folder[4])

input_tensor = Input(shape=(240,320,3))
model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False)
videoRNN = load_model('./p3.hdf5')
outcome_V1 = model.predict(valid_X1,verbose = 1)
predict_X1 = videoRNN.predict(outcome_V1.reshape((1,-1,2048)),verbose = 1)
ans_1 = np.argmax((predict_X1),axis =2)
print(ans_1)
with open(os.path.join(sys.argv[2],str(valid_folder[0])+'.txt'), 'w') as f:
    for i in ans_1[0]:
        f.write('%d\n' %(i))

outcome_V2 = model.predict(valid_X2,verbose = 1)
predict_X2 = videoRNN.predict(outcome_V2.reshape((1,-1,2048)),verbose = 1)
ans_2 = np.argmax(np.around(predict_X2),axis =2)
with open(os.path.join(sys.argv[2],str(valid_folder[1])+'.txt'), 'w') as f:
    for i in ans_2[0]:
        f.write('%d\n' %(i))

outcome_V3 = model.predict(valid_X3,verbose = 1)
predict_X3 = videoRNN.predict(outcome_V3.reshape((1,-1,2048)),verbose = 1)
ans_3 = np.argmax(np.around(predict_X3),axis =2)
with open(os.path.join(sys.argv[2],str(valid_folder[2])+'.txt'), 'w') as f:
    for i in ans_3[0]:
        f.write('%d\n' %(i))

outcome_V4 = model.predict(valid_X4,verbose = 1)
predict_X4 = videoRNN.predict(outcome_V4.reshape((1,-1,2048)),verbose = 1)
ans_4 = np.argmax(np.around(predict_X4),axis =2)
with open(os.path.join(sys.argv[2],str(valid_folder[3])+'.txt'), 'w') as f:
    for i in ans_4[0]:
        f.write('%d\n' %(i))
outcome_V5 = model.predict(valid_X5,verbose = 1)
predict_X5 = videoRNN.predict(outcome_V5.reshape((1,-1,2048)),verbose = 1)
ans_5 = np.argmax(np.around(predict_X5),axis =2)
with open(os.path.join(sys.argv[2],str(valid_folder[4])+'.txt'), 'w') as f:
    for i in ans_5[0]:
        f.write('%d\n' %(i))


import numpy as np
import os,sys
from skimage import io
from keras.layers import *
from keras.models import *

model = load_model('./FCN8.h5py')
image_path= sys.argv[1]
n_image = os.listdir(image_path)
data = np.zeros([len(n_image),512,512,3])
predit_path = sys.argv[2]
k = 0
for file in n_image:
    data[k] = io.imread(os.path.join(sys.argv[1],file))
    k+=1

p = model.predict(data,verbose=1,batch_size=2)
p = p.reshape([len(n_image),512*512,7])

img_p = np.zeros([len(n_image),512*512,1])
for i in range(len(n_image)):
    img_p[i] = (np.argmax(p[i,:],axis = 1)).reshape(-1,1)
img_p = img_p.reshape([len(n_image),512,512])
img_p[img_p>0]+=1

im = np.zeros([512,512,3],dtype = np.uint8)
for i in range(len(n_image)):
    im[:,:,0] = np.round(img_p[i]%2*255)
    im[:,:,1] = np.round((img_p[i]//2)%2*255)
    im[:,:,2] = np.round((img_p[i]//4)%2*255)
    io.imsave(os.path.join(predit_path,'{:0>4d}'.format(i)+'_mask.png'),arr = im)
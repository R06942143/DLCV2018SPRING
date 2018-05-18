import numpy as np 
import matplotlib.pyplot as plt
import sys
import os


KLD = np.load('./curve/KLD.npy')
MSE = np.load('./curve/MSE.npy')
aa = KLD
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
for i in range(2,len(KLD)-2):
    aa[i] = (KLD[i-2]+ KLD[i-1] + KLD[i] + KLD[i+1]+ KLD[i+2])/5.0
plt.title('training KLD with lambda:3e-06')
plt.xlabel('epoch')
plt.plot(range(len(KLD)),aa[:,0])

plt.subplot(1,2,2)
bb = MSE
for i in range(2,len(MSE)-2):
    bb[i] = (MSE[i-2]+MSE[i-1] + MSE[i] + MSE[i+1]+MSE[i+2])/5.0
plt.title('training MSE')
plt.xlabel('epoch')
plt.plot(range(len(MSE)),bb[:,0])

plt.savefig(os.path.join(sys.argv[2],'fig1_2.jpg'))
plt.clf()

GAN_d = np.load('./curve/gan_d.npy')
GAN_g = np.load('./curve/gan_g.npy')

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.title('discriminator loss binary_entropy')
plt.xlabel('step')
plt.plot(range(len(GAN_d[:20000])),GAN_d[:20000,0])

plt.subplot(1,2,2)
plt.title('generator loss binary_entropy')
plt.xlabel('step')
plt.plot(range(len(GAN_g[:20000])),GAN_g[:20000])

plt.savefig(os.path.join(sys.argv[2],'fig2_2.jpg'))
plt.clf()

ACGAN_d = np.load('./curve/acgan_d.npy')
ACGAN_g = np.load('./curve/acgan_g.npy')


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)

plt.title('discriminator loss binary_entropy')
plt.xlabel('step')
plt.plot(range(len(ACGAN_d)),ACGAN_d[:,4])
plt.subplot(1,2,2)
plt.title('generator loss binary_entropy')
plt.xlabel('step')
plt.plot(range(len(ACGAN_g)),ACGAN_g[:,0])

plt.savefig(os.path.join(sys.argv[2],'fig3_2.jpg'))
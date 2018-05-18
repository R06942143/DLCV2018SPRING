wget -O VAN.h5py 'https://www.dropbox.com/s/dlrqor2v4hgnqem/VAE.h5py?dl=1'
wget -O GAN.h5py 'https://www.dropbox.com/s/7b6j940up1zd3yr/gan_30000.h5py?dl=1'
wget -O ACGAN.h5py 'https://www.dropbox.com/s/yth46w4frhdet29/ACGAN.h5py?dl=1'
python VAE.py $1 $2
python GAN.py $1 $2
python ACGAN.py $1 $2
python fig.py $1 $2

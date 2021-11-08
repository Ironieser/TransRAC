# SVIP_Counting

##1. SW + QK matrix + Density map  and Counter layer
###1.1 loss1 
Density map = [b,f]  \
labels are built by normalization.
###1.2 loss2 
counter layer = [b,1] \
count =  len(labels)/2


##2.  hyperparam
>FRAME = 128\
BATCH_SIZE = 8\
random.seed(1)\
LR = 6e-6\
W1 = 5\
W2 = 1\
NUM_WORKERS = 32
##3. checkpoint
lastCkptPath = '/p300/SWRNET/checkpoint2/ckpt15_trainMAE_1.498641728136049.pt'
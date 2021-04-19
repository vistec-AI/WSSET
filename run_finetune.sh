ARGS="--dsname bbcsport --batchsize 32 --lr 1e-4 --epoch 1000  --loss triplet --treshold 20 --istransfer --loadmodel pretrained_bbcsport.h5"

python train.py $ARGS

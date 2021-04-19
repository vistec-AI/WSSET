ARGS="--dsname bbcsport --batchsize 32 --lr 1e-4 --epoch 1000  --loss gaussian --treshold 20"

python train.py $ARGS

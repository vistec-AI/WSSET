from manager import DataManager
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def get_data_from_dataset_name(ds_name,train_size,loss,batch_size):
    data = DataManager(dataset=ds_name)
    te_idx = data.TE[0]
    tr_idx = data.TR[0]
    W, P = [np.array([i.reshape(-1) for i in data.BOW_X[:]]), np.array([i.T for i in data.X[:]])]

    #get maximum BoW size of the dataset
    max =0
    for n in range(P.shape[0]):
        if P[n].shape[0]>max:
            max=P[n].shape[0]
    print(max)

    #padding each sample's BoW to be equal as the maximum BoW size and add additional BoW weights into the array
    for n in range(P.shape[0]):
        zero = np.zeros((max,301))
        W[n]=np.reshape(W[n],(W[n].shape[0],1))
        P[n]=np.hstack((P[n],W[n]))
        zero[:P[n].shape[0],:P[n].shape[1]]=P[n]
        P[n]=zero
    
    #splitting the dataset based on given test indices from WMD paper
    P = np.asarray([i for i in P[:]])
    train = P[tr_idx]
    P_te = P[te_idx]
    y_te = te_idx

    #splitting a validation set from the given train set
    P_tr,P_va,y_tr,y_va = train_test_split(train,tr_idx,test_size=train_size)

    #generating tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((P_tr,y_tr))

    if loss!="ce" and loss!="triplet": #check if supervised learning
        train_dataset = tf.data.Dataset.from_tensor_slices((P_tr,y_tr))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((P_tr,data.Y[y_tr].astype('int16')))

    train_dataset = train_dataset.shuffle(1000).batch(batch_size,drop_remainder=True)

    return P, y_te, y_tr, y_va, data.Y, train_dataset






from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def knn_evaluate(nn=10,results=None,y_test=None):
    neigh = KNeighborsClassifier(n_neighbors=nn)
    neigh.fit(results,y_test)
    distance,indices=neigh.kneighbors(results,n_neighbors=nn)
    true=0
    truew=0
    d=0
    for i in range(distance.shape[0]):
        arr=np.zeros(20)
        arr2=np.zeros(20)
        for j in range(nn-1):
            idx =int(y_test[indices[i][j+1]])
            arr[idx]=arr[idx]+1
            arr2[idx]=arr[idx]+(1/(distance+1e-12))[i][j+1]
        pred=np.argmax(arr)
        pred2=np.argmax(arr2)  
        if pred == int(y_test[i]):
            true=true+1
        if pred2 == int(y_test[i]):
            truew=true+1
    acc=true/distance.shape[0]
    acc2=truew/distance.shape[0]
    return acc2

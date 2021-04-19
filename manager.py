import os.path
import scipy.io
import numpy as np
class DataManager:
    """
    To manage kEMD classification dataset
    """
    def __init__(self, dataset):
        """
        this func loads .mat file and save into each variavle
        input: dataset name
        output: -
        """

        # path to dataset
        dataset_path = os.path.dirname(__file__) + '/data/'
        
        #dataset alias dict
        dataset_dict = {'twitter' : 'twitter-emd_tr_te_split.mat',
                        'classic' : 'classic-emd_tr_te_split.mat',
                        'recipe'  : 'recipe2-emd_tr_te_split.mat',
                        'amazon'  : 'amazon-emd_tr_te_split.mat',
                        'bbcsport': 'bbcsport-emd_tr_te_split.mat'
                        }
        
        #chance dataset alias to exact dataset name
        filename = dataset_dict.get(dataset.lower())
        
        #load dataset
        mat = scipy.io.loadmat(dataset_path+filename)
        
        #assign to variables
        self.TR, self.TE, self.words, self.BOW_X, self.X, self.Y = self.mat_to_var_split(mat=mat)

        self._EPSILON = 1e-7

    
    def mat_to_var_split(self, mat):
        """Split .mat data to each varible
        TR [5,n]: each row corresponds to a random split of the training set, 
        each entry is the index with respect to the full dataset. So for example, 
        to get the BOW of the training set for the third split do: BOW_xtr = BOW_X(TR(3,:))
        
        TE [5,ne]: same as TR except for the test set

        words [1,n+ne]: each cell corresponds to a document and is itself a {1,u} 
        cell where each entry is the actual word corresponding to each unique word

        BOW_X [1,n+ne]: each cell in the cell array is a vector corresponding to a document. 
        The size of the vector is the number of unique words in the document, 
        and each entry is how often each unique word occurs.

        X [1,n+ne]: each cell corresponds to a document and is a [d,u] matrix where 
        d is the dimensionality of the word embedding, 
        u is the number of unique words in that document, 
        n is the number of training points, and 
        ne is the number of test points.

        Y [1,n+ne]: the label of each document

        input = dict from .mat file
        output = [TR, TE, words, BOW_X, X, Y]
        """

        TR = mat['TR']-1 #minus 1 because original index is start from 1

        TE = mat['TE']-1 #minus 1 because original index is start from 1

        try:
            words = mat['words'].reshape(-1)#reshape to (n+ne,)
        except:
            words = mat['the_words'].reshape(-1) #reshape to (n+ne)

        BOW_X = mat['BOW_X'].reshape(-1)#reshape to (n+ne,)

        ## Each column is the word2vec vector for a particular word.
        X = mat['X'].reshape(-1) #position of each word #reshape to (n+ne,)

        Y = mat['Y'].reshape(-1) #reshape to (n+ne,)

        return ([TR, TE, words, BOW_X, X, Y])
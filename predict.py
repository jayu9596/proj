
import numpy as np
from numpy import random as rand
from scipy import sparse as sps
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
import os
import time as tm
# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVtst_X_Xftst_X_Xftst_X_XfIOR

# INPUT CONVENTION
# X: n x d matrix in csr_matrix format containing d-dim (sparse) features for n test data points
# k: the number of recommendations to return for each test data point in ranked order

# OUTPUT CONVENTION
# The method must return an n x k numpy nd-array (not numpy matrix or scipy matrix) of labels with the i-th row 
# containing k labels which it thinks are most appropriate for the i-th test point. Labels must be returned in 
# ranked order i.e. the label yPred[i][0] must be considered most appropriate followed by yPred[i][1] and so on

# CAUTION: Make sure that you return (yPred below) an n x k numpy (nd) array and not a numpy/scipy/sparse matrix
# The returned matrix will always be a dense matrix and it terribly slows things down to store it in csr format
# The evaluation code may misbehave and give unexpected results if an nd-array is not returned

def getReco( X, k):
    # Find out how many data points we have
    n = X.shape[0]
    L = 3400
    # Load and unpack the dummy model
    # The dummy model simply stores the labels in decreasing order of their popularity
    
    out_path="sandbox/data/Assn2/"
    
    model_dir = "sandbox/results/Assn2/"
    
    dump_food(X,out_path)
    os.system("bash shallow/sample_run.sh")
    filename = model_dir + "score_mat"
    
    Xp, _ = load_svmlight_file( "%s.txt" %filename, multilabel = True, n_features = L, offset = 1 )

    yPred = np.zeros( (n, k), dtype=int )

    for ind, user in enumerate(Xp):
        d = user.data
        i = user.indices
        xf = np.vstack( (i, d) ).T
        xf = xf[xf[:,1].argsort()[::-1]]
        for j in range(0,k):
            yPred[ind][j]=xf[j][0]
    os.system("rm sandbox/data/Assn2/tst_X_Xf.txt")
    os.system("rm sandbox/results/Assn2/score_mat.txt")
    '''
    # Let us predict a random subset of the 2k most popular labels no matter what the test point
    shortList = model[0:2*k]
    # Make sure we are returning a numpy nd-array and not a numpy matrix or a scipy sparse matrix
    yPred = np.zeros( (n, k) )
    for i in range( n ):
        yPred[i,:] = rand.permutation( shortList )[0:k]
    '''
    return yPred


def dump_food( matrix_test, out_path):
    (n, d) = matrix_test.shape
    dummy = sps.csr_matrix( (n, 1) )
    dump_svmlight_file( matrix_test, dummy, "test_data.X", multilabel = True, zero_based = True, comment = "%d %d" % (n, d) )   

    test_ws=open("test_data.X","r")
    test_is=open(out_path+"tst_X_Xf.txt","w")
    lines=test_ws.readlines()
    for i in range(0,len(lines)):
        if(lines[i][0]=='#'):
            if(len(lines[i])>2):
                if(not(lines[i][2]<='9' and lines[i][2]>='0')):
                    continue
                else :
                    lines[i]=lines[i][2:]
                    test_is.write(lines[i])
            else :
                continue
        else :
            lines[i]=lines[i][1:]
            test_is.write(lines[i])

    test_is.close()
    os.system("rm test_data.X")


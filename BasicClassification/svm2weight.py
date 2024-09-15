# Compute the weight vector of linear SVM based on the model file
# Original Perl Author: Thorsten Joachims (thorsten@joachims.org)
# Python Version: Ori Cohen (orioric@gmail.com)
# Call: python svm2weights.py svm_model

# This is heavily borrowed from http://www.cs.cornell.edu/people/tj/svm_light/svm2weight.py.txt

from operator import itemgetter
import numpy as np

def sortbyvalue(d,reverse=True):
    ''' proposed in PEP 265, using  the itemgetter this function sorts a dictionary'''
    return sorted(d.items(), key=itemgetter(1), reverse=True)

def sortbykey(d,reverse=True):
    ''' proposed in PEP 265, using  the itemgetter this function sorts a dictionary'''
    return sorted(d.items(), key=itemgetter(0), reverse=False)

def get_svmlight_weights(weights_path, printOutput=True):
    f = open(weights_path, "r")
    i=0
    lines = f.readlines()
    w = {}
    for line in lines:
        if i>10:
            features = line[:line.find('#')-1]
            comments = line[line.find('#'):]
            alpha = features[:features.find(' ')]
            feat = features[features.find(' ')+1:]
            for p in feat.split(' '): # Changed the code here. 
                a,v = p.split(':')
                if not (int(a) in w):
                    w[int(a)] = 0
            for p in feat.split(' '): 
                a,v = p.split(':')
                w[int(a)] +=float(alpha)*float(v)
        elif i==1:
            if line.find('0')==-1:
                print('Not linear Kernel!')
                printOutput = False
                break
        elif i==10:
            if line.find('threshold b')==-1:
                print('Parsing error!')
                printOutput = False
                break 
            else:
                thresh = float(line.split('#')[0])
        i+=1    
    f.close()
    ws = sortbykey(w)
    if printOutput:
        for (i,j) in ws:
            print(f"{i}:{j}")
            i+=1
    ws_vec = np.array([x[1] for _,x in enumerate(ws)]).reshape(-1,1)
    return ws_vec, thresh
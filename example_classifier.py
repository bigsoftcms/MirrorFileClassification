#!/usr/bin/env python
# 
#  This is an example use of Brandon't classifler
#  You need to run example_training.py on the training files first,
#  in order to create the model file xgb_model.pkl which is used
#  by this program
#
#  Usage:    example_classifier.py  <file> [ <file>  ... ]
#
#  outputs list of calls
#

import FileClassification
import pickle
import sys
import xgboost as xgb
import torch
import time
import numpy as np
import pandas as pd

from pprint import pprint

prefix_length = 3000                                # 3kb used for the model
filetype = ['fna', 'gbff', 'gff', 'fastq', 'sra']   # this order is set in FileClassifier.trainloader:

if len( sys.argv ) < 2:
   print( "please provide a test file on the command line." )
   quit()

# Tests Classifier. Returns array of predicted classes

#  Brandon's code included a list of threshold cutoffs for score
#  for analysis I believe.  I've left this in for now, but maybe
#  it should be removed to simply things

def ExampleFileClassifier( model, data_test, threshold ):
   
    data_test = xgb.DMatrix(data_test)
   
    for thperc in threshold:    # this loop can be removed I think
        preds_threshold = []

        preds = model.predict(data_test)
   
        # preds is an array of length N where N is the number of query
        # files.   Each element is an array of length five, for the
        # five file types, and each value is the score the classifier
        # assigns for the file being that type.  Range is 0 to 1.

        # next part extracts indices of highest score for each
        # query file
       
        best_preds = []
        below_ind = []

        for p in range(len(preds)):
            if np.max(preds[p]) < thperc:
                below_ind.append(p)
               
            else:
                preds_threshold.append(preds[p])
                best_preds.append(np.argmax([preds[p]]))
       
        best_preds = [ filetype[i] for i in np.array(best_preds) ]
 
    return best_preds



                               ################
                               # Main program #
                               ################

query_files = sys.argv[1:]

# load previously trained model

xgb_model = pickle.load( open( "xgb_class.pkl", "rb") )

#
# collect rkb strings from each file
#
prefixes = []  # list of first 3kb strings from each file

for query_file in query_files:
    with open( query_file, "rb" ) as qf:
        prefixes.append( qf.read(prefix_length).decode("utf-8") )

#
# convert to character vectors (n-grams)
#

vec = FileClassification.test_char_vectorizer( pd.Series( prefixes ) )

# Make the classifier calls

pred = ExampleFileClassifier( xgb_model, vec, [0.7] )

# print 'em out along with the filenames

for i in range( 0, len(pred) ):
    print( "{0} {1}".format( pred[i], query_files[i] ) )
    
print( "done." )




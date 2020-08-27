#!/usr/bin/env python
#
#   Example training script.
#
#   Inputs are directories of training files, one type per directory.
#   These are coded below.
#
#   This creates two output files, both pickle format
#
#      tfidfcv2.pkl  - character vectorizing.  This is basically a vector of kmers
#                      determined for each file class or type (I believe)
#                      A logic flag below allows the generation of this to be suppressed
#                      and instead the vectors are loaded from the existing file
#
#      xgb_model.pkl - the ML model. This can also be loaded instead of computed and saved
#                      with a logical flag.
#
import FileClassification

from pprint import pprint

print( "Oh Hai" )

datadir = '../Data/'

training_file_names =  ['genomic.fna',
                        'genomic.gbff',
                        'genomic.gff',
                        'FASTQ_truncated',    # There were two rounds of data collection.
                        'TruncFASTQ10000',    # Brandon included orignal prototype file sets for 
                        'SRA_truncated',      # FASTQ and SRA for good measure.
                        'TruncSra10000'
                       ]

training_paths = [ datadir + tf for tf in training_file_names ]

pprint( training_paths )


# The function FileClassification.trainloader makes the type-to-file assignment based 
# on filename extension.  (If new file types are added, that section will have to be
# adjusted by some means.)
#
# also, in principle the type array here could be derived from the file types,
# although this still may be necessary to determine an order in the called score vectors.
#

filetype = ['fna', 'gbff', 'gff', 'fastq', 'sra']   # this order is set in "trainloader:

filelength = 3000 # bytes we want to analyze from each file

print( "train loading" )
data_iterable = FileClassification.trainloader( training_paths, filetype, filelength )
#pprint( data_iterable )

param = { # XGB Parameters
    'max_depth': 10,  # the maximum depth of each tree
    'eta': 0.1,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': len(filetype),  # the number of classes that exist in this datset
    'booster' : 'dart', # Dropout added
    'rate_drop' : 0.4, #Dropout Rate
    'skip_drop' : 0.4, # Probability of skipping Dropout
    } 
num_round = 40 # Number of rounds we train our XGB Classifier
ngram_range = (4,8) # Character length we search by for Tfidf
max_features = 12 # Vocab Words Per File Class
Dataset = data_iterable
ttsplit = 0.2  # Train/Test Split Percentage

# Data is split into Train/Validation/Test. Stratified sampling
print( "splitting..." )
X_train, X_val, X_test, y_train, y_val, y_test = FileClassification.Data_Splitter(Dataset, ttsplit)
print( type( X_train ) )
print( dir( X_train ) )
# X_train is a pandas series
zot = X_train.to_numpy() 
print( type( zot ) )
print( dir( zot ) )
print( zot.shape )
pprint( zot[38000] )
#pprint( zot )

#pprint( X_train )
quit()


# Counts and Percentages of each class in each data split

FileClassification.classcounts([y_train,y_val,y_test],5, filetype)

# Train and Validation Set are transformed via TF-IDF Vectorizer

# change load=False to generate the vector file "tfidfcv2.pkl" the first time
# change to load=True for subsequent runs (saves time)

dat_train, dat_val = FileClassification.Char_vectorizer2(X_train, y_train, X_val, y_val, filetype, ngram_range, max_features, load = False)   


# XGBoost Classifier Training

xgbmodel = FileClassification.TrainXGBClassifier(param, num_round, dat_train, dat_val, y_train, y_val, load = False)

print( "Done.  Training complete." )


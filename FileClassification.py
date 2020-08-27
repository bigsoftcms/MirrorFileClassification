
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import re
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_curve, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
import os
import gzip
import random
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
import pickle
import seaborn as sns
import scikitplot as skplt
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch.utils.data as td
from tqdm.autonotebook import tqdm, trange
import inspect
import warnings
warnings.filterwarnings('ignore')


# Returns a string of length bytes long
def dataloader(filepath, length):
    f = gzip.GzipFile(fileobj=open(filepath, 'rb'))
    data = f.read(length)
    return data.decode("utf-8")



# Returns dataframe of # of files long. Analyze the first length bytes of each file
def trainloader(direclist, filetype, length):
    fileid = np.arange(len(filetype))
    dat = []
    filelabels = []
    c = -1
    for direc in direclist:
        c+=1
        for file in os.listdir(direc):
            if file.endswith('fna.gz'):
                tempdat = dataloader(os.path.join(direc,file),length)
                dat = dat + [tempdat]
                filelabels.append(0)
            if file.endswith('fna.trunc'):
                tfile = open(os.path.join(direc,file), 'rb')
                tempdat = tfile.read(length).decode("utf-8")
                dat = dat + [tempdat]
                filelabels.append(0)
                
            if file.endswith('gbff.gz'):
                tempdat = dataloader(os.path.join(direc,file),length)
                dat = dat + [tempdat]
                filelabels.append(1)
            if file.endswith('gbff.trunc'):
                tfile = open(os.path.join(direc,file), 'rb')
                tempdat = tfile.read(length).decode("utf-8")
                dat = dat + [tempdat]
                filelabels.append(1)
                
            if file.endswith('gff.gz'):
                tempdat = dataloader(os.path.join(direc,file),length)
                dat = dat + [tempdat]
                filelabels.append(2)
            if file.endswith('gff.trunc'):
                tfile = open(os.path.join(direc,file), 'rb')
                tempdat = tfile.read(length).decode("utf-8")
                dat = dat + [tempdat]
                filelabels.append(2)
                
            if file.endswith('fastq.truncated'):
                tfile = open(os.path.join(direc,file), 'rb')
                tempdat = tfile.read(length).decode("utf-8")
                dat = dat + [tempdat]
                filelabels.append(3)
            if file.endswith('sra.truncated'):
                tfile = open(os.path.join(direc,file), 'rb')
                tempdat = tfile.read(length).decode("ISO-8859-1")
                dat = dat + [tempdat]
                filelabels.append(4)
                
    data = {'Data': dat,
        'Type': filelabels
    }
    df = pd.DataFrame(data, columns = ['Data', 'Type'])
    return(df)


# Returns rows of correctly classified features

def rowextraction(predicted, truth):
    return([np.where(predicted==truth)[0], np.where(predicted!=truth)[0]])


# Returns distribution of probability of class being correct for correctly classified features

def uncertainty(pred, row_ind, num_classes):
    class_arr = [[] for i in range(num_classes)]
    dist_arr = []
    for ind in row_ind:
        class_ind = np.argmax(pred[ind])
        class_arr[class_ind].append(np.max(pred[ind]))
    for arr in class_arr:
        dist_arr.append([np.mean(arr),np.std(arr)])
    return(dist_arr)


# Returns Graphs of Feature Importance by Importance Type

def FeatureImportance(model, features, importance):
    for im in importance:
        feature_important = model.get_score(importance_type = im)
        keys = list(feature_important.keys())
        feature_chars = []
        keylist = []
        for key in keys:
            keylist.append(int(key.replace('f','')))
        for i in keylist:
            feature_chars.append(features[i])
        values = list(feature_important.values())
        fdata = pd.DataFrame(data=values, index=feature_chars, columns=["score"]).sort_values(by = "score", ascending=False)
        fdata.plot(kind='barh', title = im)    
        

# Prints Confusion Matrix

def confusionmatrix(truth, pred, names):
    cm = confusion_matrix(truth, pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar().set_label("# Classified")
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.show()



# Returns Accuracy Per Class

def classaccuracy(truth, pred, ind, num_classes):
    classes = np.arange(num_classes)
    predcl = np.asarray([np.argmax(pred[i]) for i in ind])
    acc = []
    for cl in classes:
        truth_len = len(np.where(truth == cl)[0])
        pred_len = len(np.where(predcl == cl)[0])
        acc.append(pred_len/truth_len)
    return(acc)



#Splits our data into train/test

def Data_Splitter(Dataset, ttsplit):
    X = Dataset.iloc[:,0]
    y = Dataset.iloc[:,1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ttsplit , stratify = y, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=ttsplit , stratify = y_train, random_state = 42)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Tf-idf Transformation on Training and Validation data based on max_features # of vocabulary words
# generated from each file class.

def Char_vectorizer2(X_train, y_train, X_test, y_test, filetype, ngram_range, max_features, load):
    
    fitset = trainselector2(np.array(X_train), np.array(y_train), len(filetype))
    
    if load == False:
        start = time.time()
        print("Generating Character Vectorizer...")
        char_vocab = {}
        for fileset in fitset:
            char_vectorizer = TfidfVectorizer(analyzer = 'char',
                ngram_range = ngram_range, max_features = max_features)
            char_vectorizer.fit(fileset)
            char_vocab.update(char_vectorizer.vocabulary_)
            
        count = -1
        for key in char_vocab.keys():
            count+=1
            char_vocab[key] = count
            
    
        char_vectorizer2 = TfidfVectorizer(analyzer = 'char',

                ngram_range = ngram_range, max_features = max_features,
                                vocabulary = char_vocab)
        char_vectorizer2.fit(X_train)
        print(char_vectorizer2.get_feature_names())
        
        train_chars = char_vectorizer2.transform(X_train)
        test_chars = char_vectorizer2.transform(X_test)
        pickle.dump(char_vectorizer2, open("tfidfcv2.pkl", "wb"))
        print("Character Vectorizer Saved")
        end = time.time()
        print("Vectorizer Train Time: %d" % (end-start))
        
    if load == True:
        start = time.time()
        print("Loading Character Vectorizer...")
        char_vectorizer = pickle.load(open("tfidfcv2.pkl", 'rb'))
        print("Character Vectorizer Loaded")
    
        print(char_vectorizer.get_feature_names())
        train_chars = char_vectorizer.transform(X_train)
        test_chars = char_vectorizer.transform(X_test)
        end = time.time()
        print("Vectorizer Load Time: %d" % (end-start))
    
    
    
    return train_chars, test_chars



# Prints amount of files per class and percentage representation of each class per set

def classcounts(y_set, nclass, filetypes):
    setnames = ['Train', 'Validation', 'Test']
    ind = -1
    print("Filetype Location: {}".format(filetypes))
    for y in y_set:
        ind+=1
        y = np.asarray(y)
        counts = [[] for i in range(nclass)]
        percs = [[] for i in range(nclass)]
        labels = np.arange(nclass)
        for c in range(nclass):
            local = len(np.where(y == c)[0])
            counts[c] = local
            percs[c] = round((local/len(y)),4)
        print(setnames[ind] + ' counts: {}'.format(counts))
        print(setnames[ind] + ' percentages: {}'.format(percs))
        

# Vectorizes test set and converts to DMatrix

def test_char_vectorizer(X_test):
    start = time.time()
    #print("Loading Character Vectorizer...")
    char_vectorizer = pickle.load(open("tfidfcv2.pkl", 'rb'))
    #print("Character Vectorizer Loaded")
    #print(char_vectorizer.get_feature_names())
    
    test_chars = char_vectorizer.transform(X_test)
    
    #dtest = xgb.DMatrix(test_chars)
    end = time.time()
    #print("Vectorizer Load Time: %d" % (end-start))
    
    return test_chars


# Timing Simulation for 1 file

def FileVecTime(X_test, Vectorizer, nsims, nfile, model):
    Onefiletimes = []
    for i in range(nsims):
        r = np.random.randint(len(X_test))
        localtest = X_test.iloc[r]
        Start_t = time.time()
        vectest = Vectorizer([localtest])
        data_test = xgb.DMatrix(vectest)
        pred = model.predict(data_test)
        End_t = time.time()
        
        Onefiletimes.append(End_t-Start_t)
    
    return(np.mean(Onefiletimes), np.std(Onefiletimes))
        
        
        
# Splits our training data into parts by filetype

def trainselector2(train_data, train_label, n_class):
    labelmat = [[] for i in range(n_class)]
    
    for c in range(n_class):
        t_labelloc = np.where(train_label == c)[0]
        labelmat[c].extend(train_data[t_labelloc])
    
    return(labelmat)
        

# Trains/Loads XGB Classifier

def TrainXGBClassifier(param, num_round, train_dat, val_dat, y_train, y_val, load):
    train_dat = xgb.DMatrix(train_dat, label = y_train)
    val_dat = xgb.DMatrix(val_dat, label = y_val)
    if load == False:
        start = time.time()
        print("Training Model...")
        model = xgb.train(param, train_dat, num_round, evals = [(train_dat, 'train'), (val_dat, 'eval')], verbose_eval = True)
        pickle.dump(model, open("xgb_class.pkl", "wb"))
        print("Model Saved")
        end = time.time()
        print('Training time: %d' % (end-start))
    if load == True:
        print("Loading Model...")
        model = pickle.load(open("xgb_class.pkl", "rb"))
        print("Model Loaded")
        
    return model


# Trains/Loads SVM Classifier

def TrainSVMClassifier(train_dat, y_train, load):
    if load == False:
        start = time.time()
        print("Training Model...")
        model = OneVsRestClassifier(SVC(kernel='linear', probability=True, C=1))
        model.fit(char_train, y_train)
        pickle.dump(model, open("svm_class.pkl", "wb"))
        print("Model Saved")
        end = time.time()
        print('Training time: %d' % (end-start))
    if load == True:
        print("Loading Model...")
        model = pickle.load(open("svm_class.pkl", "rb"))
        print("Model Loaded")
        
    return model


# File Classification Multilayer Perceptron

class FileNet(nn.Module):
    def __init__(self):
        super(FileNet, self).__init__()
        self.fc1 = nn.Linear(60, 90)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(90, 120)
        self.fc3 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, p = 0.3, training = True)
        x = self.fc2(x)
        x = self.relu(x)
        x = F.dropout(x, p = 0.3, training = True)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = F.softmax(x, dim = 1)
        
        return x


# Metric Calculation and Visualization for MLP Source: https://github.com/marrrcin/pytorch-resnet-mnist

def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")


# Trains/Loads MLP Classifier

def TrainMLPClassifier(Net, train_dat, y_train, val_dat, y_val, epochs, load):
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    
    train_x = torch.Tensor(train_dat.toarray())
    train_y = torch.Tensor(y_train).long()
    train_ds = utils.TensorDataset(train_x,train_y)
    train_loader = td.DataLoader(train_ds, batch_size=10,
        shuffle=False, num_workers=1)

    val_x = torch.Tensor(val_dat.toarray())
    val_y = torch.Tensor(y_val).long()
    val_ds = utils.TensorDataset(val_x,val_y)
    val_loader = td.DataLoader(val_ds, batch_size=10,
        shuffle=False, num_workers=1)
    print("Loaders ready")
    
    if load == False:
        batches = len(train_loader)
        val_batches = len(val_loader)
        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_criteria = nn.CrossEntropyLoss()
        start = time.time()
        print("Training Model...")
        for epoch in range(epochs):
            total_loss = 0
            progress = tqdm(enumerate(train_loader), desc = "Loss: ", total = batches)
            model.train()
            for batch, tensor in progress:
                data, target = tensor
                optimizer.zero_grad()
                out = model(data)
                loss = loss_criteria(out, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress.set_description("Loss: {:.4f}".format(total_loss/(batch+1)))
                
            val_losses = 0
            precision, recall, f1, accuracy = [], [], [], []

            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    X, y = data
                    outputs = model(X)

                    val_loss = loss_criteria(outputs, y)
                    val_losses += val_loss.item()

                    predicted_classes = torch.max(outputs.data, 1)[1]

                    for acc, metric in zip((precision, recall, f1, accuracy), 
                                           (precision_score, recall_score, f1_score, accuracy_score)):
                        acc.append(
                            calculate_metric(metric, y, predicted_classes)
                        )


            print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
            print_scores(precision, recall, f1, accuracy, val_batches)
            #losses.append(total_loss/batches)
        
        torch.save(model.state_dict(), 'FileClassMLP.pt')
        print("Model Saved")
        end = time.time()
        print('Training time: %d' % (end-start))
        
    if load == True:
        print("Loading Model...")
        model = Net()
        model.load_state_dict(torch.load('FileClassMLP.pt'))
        print("Model Loaded")
        
    return model
        

# Tests Classifier

def TestFileClassifier(model, data_test, filetype, y_test, output, threshold, threshold_plots, classifier):
      
    global_precision = []
    global_recall = []
    global_F1 = []
    global_class = [[] for c in range(len(filetype))]
    global_retain = []
    
    if classifier == 'xgb':
        data_test = xgb.DMatrix(data_test)
    if classifier == 'mlp':
        data_test = torch.Tensor(data_test.toarray())
    
    for thperc in threshold:
        y_test = np.asarray(y_test)
        
        y_test_threshold = []
        preds_threshold = []
        start = time.time()
        if classifier == 'xgb':
            preds = model.predict(data_test)
        if classifier == 'svm':
            preds = model.predict_proba(data_test)
        if classifier == 'mlp':
            model.eval()
            with torch.no_grad():
                preds = model(data_test).data.numpy()
        end = time.time()
        
        best_preds = []
        belowthresh = 0
        below_ind = []
        count = -1
        for p in range(len(y_test)):
            if np.max(preds[p]) < thperc:
                belowthresh += 1
     
                below_ind.append(p)
                
            else:
                y_test_threshold.append(y_test[p])
                preds_threshold.append(preds[p])
                best_preds.append(np.argmax([preds[p]]))
        
        best_preds = np.array(best_preds)
        y_test_threshold = np.array(y_test_threshold)
        new_preds = np.delete(preds, below_ind,0)
        
        local_retain = len(best_preds)/len(y_test)
        global_retain.append(local_retain)
        
        local_precision = precision_score(y_test_threshold, best_preds, average='macro')
        local_recall = recall_score(y_test_threshold, best_preds, average = 'macro')
        local_F1 = 2*(local_precision*local_recall)/(local_precision + local_recall)
        global_precision.append(local_precision)
        global_recall.append(local_recall)
        global_F1.append(local_F1)
        
        Y_bin = label_binarize(y_test_threshold, classes = [*range(len(filetype))])
        curr_thresh = thperc
        
        prec = dict()
        rec = dict()
        for f in range(len(filetype)):
            prec[f],rec[f],_ = precision_recall_curve(Y_bin[:,f], new_preds[:,f])
            plt.plot(rec[f], prec[f], lw = 2, label = 'class '+filetype[f])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.title("Precision vs. Recall Curve for Threshold: {}".format(curr_thresh))
        plt.figure(figsize = (20,10))
        plt.show()
        
        
        fpr = dict()
        tpr = dict()
        for e in range(len(filetype)):
            fpr[e], tpr[e], _ = roc_curve(Y_bin[:, e],
                                          new_preds[:, e])
            plt.plot(fpr[e], tpr[e], lw=2, label='class '+filetype[e])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="best")
        plt.title("ROC Curve for Threshold: {}".format(curr_thresh))
        plt.show()

        
        if output == True:
            print('Testing time: %d' % (end-start))
            
            if classifier == 'xgb':
                char_vectorizer = pickle.load(open("tfidfcv2.pkl", 'rb'))
                FeatureImportance(model, char_vectorizer.get_feature_names(), ['weight','cover','gain'])
            
            print("Precision: {}".format(local_precision))
            print("Recall: {}".format(local_recall))
            print("F1: {}".format(local_F1))
            
            y_test_arr = np.asarray(y_test_threshold).reshape(1,-1).squeeze()
            correct_rows = rowextraction(best_preds,y_test_arr)[0]
            incorrect_rows = rowextraction(best_preds,y_test_arr)[1]
            class_acc = classaccuracy(y_test_arr, preds_threshold, correct_rows, len(filetype))
            pred_uncertainty = uncertainty(preds, correct_rows, len(filetype))
            for i in range(len(filetype)):
                global_class[i].append(round(class_acc[i],2))
                
            for i in range(param['num_class']):
                print("Class {}".format(filetype[i]) + " mean and sd: {}".format(pred_uncertainty[i])
                     + ", Accuracy: {}".format(round(class_acc[i],2)))
            confusionmatrix(y_test_arr, best_preds, filetype)
            skplt.metrics.plot_roc(y_test_arr, new_preds)
            plt.show()

    if threshold_plots == True:
        fig, ax = plt.subplots()

        ax.plot(threshold, global_precision)
        ax.set(xlabel = '% Confident Threshold', ylabel = 'Precision', title = 'Threshold vs Precision')
        ax.grid()
        plt.figure(figsize = (20,10))

        plt.show()
        
        fig, ax = plt.subplots()

        ax.plot(threshold, global_recall)
        ax.set(xlabel = '% Confident Threshold', ylabel = 'Recall', title = 'Threshold vs Recall')
        ax.grid()
        plt.figure(figsize = (20,10))

        plt.show()
        
        fig, ax = plt.subplots()

        ax.plot(threshold, global_F1)
        ax.set(xlabel = '% Confident Threshold', ylabel = 'F1', title = 'Threshold vs F1')
        ax.grid()
        plt.figure(figsize = (20,10))

        plt.show()
        
        fig, ax = plt.subplots()

        ax.plot(threshold, global_retain)
        ax.set(xlabel = '% Confident Threshold', ylabel = '% Retained', title = 'Threshold vs Retained')
        ax.grid()
        plt.figure(figsize = (20,10))

        plt.show()
        
        
        
    if threshold_plots == True and output == True:
        for cl in range(len(filetype)):
            figc, axc = plt.subplots()
            axc.plot(threshold, global_class[cl])
            axc.set(xlabel = '% Confident Threshold', ylabel = filetype[cl] + ' accuracy', title = 'Threshold vs ' + filetype[cl])
            axc.grid()
            plt.figure(figsize = (20,10))

            plt.show()

  
    return best_preds


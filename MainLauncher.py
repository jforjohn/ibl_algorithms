##
from MyPreprocessing import MyPreprocessing
from MyIBL import MyIBL
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from config_loader import load
import argparse
import sys
from os import walk

import numpy as np
from time import time
import matplotlib.pyplot as plt

def getProcessedData(path, dataset, filename):
    try:
        Xtrain, meta = loadarff(path + filename)
    except FileNotFoundError:
        print("Dataset '%s' cannot be found in the path %s" % (dataset, path))
        sys.exit(1)

    # Preprocessing
    preprocess = MyPreprocessing()
    preprocess.fit(Xtrain)
    df = preprocess.new_df
    labels = preprocess.labels_
    return df, labels

##
if __name__ == '__main__':
    ##
    # Loads config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="ibl.cfg",
        help="specify the location of the clustering config file"
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    config = load(config_file)

    dataset = config.get('ibl', 'dataset')
    path = 'datasetsCBR/' + dataset + '/' #+ dataset + '.fold.00000'
    filenames = [filename for _,_,filename in walk(path)][0]

    train_obj = []
    for f_ind in range(0,len(filenames), 2):
        test_file = filenames[f_ind]
        train_file = filenames[f_ind+1]

        df_train, ytrain = getProcessedData(path, dataset, train_file)
        df_test, ytest = getProcessedData(path, dataset, test_file)

        clf = MyIBL(3)
        clf.fit(df_train, ytrain)
        train_obj.append(clf)
        pred = clf.predict(df_test, ytest)
        break
print(clf.classificationTest)
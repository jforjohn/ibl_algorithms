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
from itertools import product

import numpy as np
from time import time
import matplotlib.pyplot as plt

def getProcessedData(path, dataset, filename, raw=False):
    try:
        Xtrain, meta = loadarff(path + filename)
    except FileNotFoundError:
        print("Dataset '%s' cannot be found in the path %s" % (dataset, path))
        sys.exit(1)

    # Preprocessing
    preprocess = MyPreprocessing(raw)
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
    n_neighbors = [3, 5, 7]
    distances = ['euclidean', 'cosine', 'canberra']
    votings = ['mvs', 'mp', 'brd']
    combinations = [n_neighbors,
                    distances,
                    votings]

    results = pd.DataFrame()
    for n_neighbor, distance, voting in product(*combinations):
        accuracy = 0
        missclassification_rate = 0
        cd_len = 0
        start = time()
        for f_ind in range(0,len(filenames), 2):
            test_file = filenames[f_ind]
            train_file = filenames[f_ind+1]
            # raw = False
            df_train, ytrain = getProcessedData(path, dataset, train_file)
            df_test, ytest = getProcessedData(path, dataset, test_file)

            clf = MyIBL(n_neighbors=n_neighbor,
                        ibl_algo='ib2',
                        voting=voting,
                        distance=distance
                        )
            clf.fit(df_train, ytrain)
            train_obj.append(clf)
            pred = clf.predict(df_test, ytest)

            size_fold = df_test.shape[0]
            cd_len += len(clf.cd) / df_train.shape[0]
            accuracy += clf.classificationTest['correct'] / size_fold
            missclassification_rate += clf.classificationTest['incorrect'] /size_fold

        duration = time() - start
        row_name = 'k=' + str(n_neighbor) + '/' + distance + '/' + voting
        df = pd.DataFrame([[duration, accuracy/10, missclassification_rate/10, cd_len/10]],
                          index=[row_name],
                          columns=['Time', 'Accuracy', 'MisclassRate', 'CDpercentage'])
        results = pd.concat([results, df], axis=0)
        print(results)

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
from scipy.stats import friedmanchisquare
from Weight_Function import feature_select

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
    labels_fac = preprocess.labels_fac
    return df, labels, labels_fac

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

    df_results = pd.DataFrame()
    accum_acc_lst = []
    accum_time_lst = []
    row_names = []
    for n_neighbor, distance, voting in product(*combinations):
        #accuracy = 0
        acc_lst = []
        time_lst = []
        missclassification_rate = 0
        cd_len = 0
        start = time()
        for f_ind in range(0,len(filenames), 2):
            test_file = filenames[f_ind]
            train_file = filenames[f_ind+1]
            # raw = False
            df_train, ytrain, ytrain_fac = getProcessedData(path, dataset, train_file)
            df_test, ytest, _ = getProcessedData(path, dataset, test_file)

            # feature selection (weight)
            df_train, df_test = feature_select(X_train=df_train, y_train=ytrain_fac, X_test=df_test, method="univariate")

            if df_train.shape[1] != df_test.shape[1]:
                missing_cols = set(df_train.columns) - set(df_test.columns)
                if not missing_cols:
                    missing_cols = set(df_test.columns) - set(df_train.columns)
                    for col in missing_cols:
                        df_train[col] = np.zeros([df_train.shape[0],1])
                else:
                    for col in missing_cols:
                        df_test[col] = np.zeros([df_test.shape[0],1])

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
            acc = clf.classificationTest['correct'] / size_fold
            acc_lst.append(acc)
            duration = time() - start
            time_lst.append(duration)
            missclassification_rate += clf.classificationTest['incorrect'] /size_fold

        row_name = 'k=' + str(n_neighbor) + '/' + distance + '/' + voting
        row_names.append(row_name)
        accum_acc_lst.append(acc_lst)
        accum_time_lst.append(time_lst)
        duration_time = sum(time_lst)
        accuracy = sum(acc_lst)
        df = pd.DataFrame([[duration_time/10, accuracy/10, missclassification_rate/10, cd_len/10]],
                          index=[row_name],
                          columns=['Time', 'Accuracy', 'MisclassRate', 'CDpercentage'])
        df_results = pd.concat([df_results, df], axis=0)
        print(df_results)

    df_acc = pd.DataFrame(accum_acc_lst, index=row_names)
    df_time = pd.DataFrame(accum_time_lst, index=row_names)
    print('Accuracy:')
    print(df_acc)
    print()
    stat, p = friedmanchisquare(*accum_acc_lst)
    print(stat, p)
    print()

    print('Time:')
    print(df_time)
    print()
    stat, p = friedmanchisquare(*accum_time_lst)
    print(stat, p)
    print()

    print('Results:')
    print(df_results.values.tolist())
    print('Accuracy')
    print(accum_acc_lst)
    print('Time')
    print(accum_time_lst)



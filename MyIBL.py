import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from random import randint


class MyIBL:
    def __init__(self, n_neighbors=1, ibl_algo='ib1', weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.ibl_algo = ibl_algo.lower()

    def similarity(self, X, inst, cd):
        sim = []
        for ind in cd:
            sim.append((-np.linalg.norm(X[ind] - inst), ind))
        return sim

    def acceptable(self, classificationRecord_ind, class_inst_counter, counter):
        if classificationRecord_ind:
            cr_correct = classificationRecord_ind['correct']
        else:
            return False

        #class_accuracy = cr_correct / (counter+1)
        correct_class_accuracy_interval = proportion_confint(cr_correct, counter+1)
        class_freq_interval = proportion_confint(class_inst_counter, counter+1)
        print('A', correct_class_accuracy_interval, class_freq_interval)
        #rel_freq = classCounter_ind / (counter+1)
        #if class_accuracy > rel_freq:

        # If the accuracy interval's lower
        # endpoint is greater than the class frequency interval's higher endpoint, then the instance
        # is accepted.
        if correct_class_accuracy_interval[0] > class_freq_interval[1]:
            print('ATrue')
            return True
        else:
            return False

    def dropInstanceFromCD(self, correct_classificationRecord_ind, class_inst_counter, counter):
        # instances are dropped when their accuracy interval's higher endpoint
        # is less than their class frequency interval's lower endpoint.
        correct_class_accuracy_interval = proportion_confint(correct_classificationRecord_ind, counter + 1)
        class_freq_interval = proportion_confint(class_inst_counter, counter + 1)
        print('D', correct_class_accuracy_interval, class_freq_interval)
        if correct_class_accuracy_interval[1] < class_freq_interval[0]:
            # drop
            print('DTrue')
            return True
        else:
            return False

    def fitIB1(self, X, y):
        cd = [0]
        for ind in range(1, X.shape[0]):
            sim = self.similarity(X, X[ind], np.array(cd))
            sim_max = max(sim)

            #print(X[ind], y[ind])
            #print(X[sim_max[1]], y[sim_max[1]])
            #print()
            if y[ind] == y[sim_max[1]]:
                self.classification['correct'] += 1
            else:
                self.classification['incorrect'] += 1
                self.misclassified.append(ind)

            cd.append(ind)
        self.cd = cd

        #print(self.classification)
        #print(X[self.misclassified])
        #print(y[self.misclassified])

    def fitIB2(self, X, y):
        cd = [0]
        for ind in range(1, X.shape[0]):
            sim = self.similarity(X, X[ind], np.array(cd))
            sim_max = max(sim)

            #print(X[ind], y[ind])
            #print(X[sim_max[1]], y[sim_max[1]])
            #print()
            if y[ind] == y[sim_max[1]]:
                self.classification['correct'] += 1
            else:
                self.classification['incorrect'] += 1
                self.misclassified.append(ind)

                cd.append(ind)
        self.cd = cd

        #print(cd)
        #print(self.classification)
        #print(X[self.misclassified])
        #print(y[self.misclassified])

    def fitIB3(self, X, y):
        cd = [0]
        classCounter = {}
        classCounter[y[0]] = 1
        classificationRecord = {}
        th_lower = 1
        th_upper = 0

        for ind in range(1, X.shape[0]):

            if not classCounter.get(y[ind]):
                classCounter[y[ind]] = 1
            else:
                classCounter[y[ind]] += 1

            # sim => (max sim, corresponding saved instance in cd)
            sim = self.similarity(X, X[ind], np.array(cd))

            sim_acceptable = []
            for cd_elem in cd:
                class_rec_cdInd = classificationRecord.get(cd_elem)
                #class_coutner_total = np.count_nonzero(y == y[cd_elem]
                class_inst_counter = classCounter[y[cd_elem]]
                if self.acceptable(class_rec_cdInd,
                    class_inst_counter,
                    ind):
                    sim_acceptable.append(sim[cd.index(cd_elem)])
            if sim_acceptable:
                sim_max_tup = max(sim_acceptable)
            else:
                i = randint(0, len(cd)-1)
                sim_max_tup = sim[i]

            sim_max = sim_max_tup[0]
            cdInst_max = sim_max_tup[1]
            if y[ind] == y[cdInst_max]:
                    self.classification['correct'] += 1
            else:
                self.classification['incorrect'] += 1
                self.misclassified.append(ind)
                cd.append(ind)
                sim = self.similarity(X, X[ind], np.array(cd))

            cd_copy = cd.copy()
            sim_copy = sim.copy()
            for cd_ind in range(len(cd)):

                if sim[cd_ind][0] >= sim_max:

                    saved_cd = cd[cd_ind]
                    if not classificationRecord.get(saved_cd):
                        classificationRecord[saved_cd] = {'correct': 0,
                                                          'incorrect': 0}
                    if y[ind] == y[saved_cd]:
                        classificationRecord[saved_cd]['correct'] += 1
                    else:
                        classificationRecord[saved_cd]['incorrect'] += 1

                    '''
                    if ((classificationRecord[saved_cd]['incorrect'] > th_lower or
                        classificationRecord[saved_cd]['correct'] < th_upper) and
                        len(cd_copy)>1):
                    '''
                    class_rec_cdInd = classificationRecord[saved_cd]['correct']
                    # class_coutner_total = np.count_nonzero(y == y[cd_elem]
                    class_inst_counter = classCounter[y[saved_cd]]
                    if self.dropInstanceFromCD(class_rec_cdInd,
                        class_inst_counter,
                        ind):
                        cd_copy.remove(saved_cd)
                        sim_copy.remove(sim[cd_ind])
            cd = cd_copy
        self.cd = cd
        print(self.cd)

    def fit(self, dataX, datay):
        if isinstance(dataX, pd.DataFrame):
            X = dataX.values
        elif isinstance(dataX, np.ndarray):
            X = dataX
        else:
            raise Exception('dataX should be a DataFrame or a numpy array')

        if isinstance(datay, pd.DataFrame):
            y = datay.values
        elif isinstance(datay, np.ndarray):
            y = datay
        else:
            raise Exception('datay should be a DataFrame or a numpy array')

        self.classification = {'correct': 0,
                               'incorrect': 0}
        self.misclassified = []

        if self.ibl_algo == 'ib1':
            self.fitIB1(X, y)
        elif self.ibl_algo  == 'ib2':
            self.fitIB2(X, y)
        elif self.ibl_algo == 'ib3':
            self.fitIB3(X, y)


    def predict(self, X):
        pass

data = np.array([[11,13],
                 [2,3],
                 [3,5],
                 [10,12],
                 [12,10],
                 [1,4]])

y = np.array([1,1,1,2,2,2])

neigh = MyIBL(1, 'ib3')
neigh.fit(data, y)

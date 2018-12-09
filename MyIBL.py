##
import numpy as np
import pandas as pd
from collections import Counter
from statsmodels.stats.proportion import proportion_confint

from random import randint
from collections import Counter



class MyIBL:
    def __init__(self, n_neighbors=1, ibl_algo='ib2', weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.ibl_algo = ibl_algo.lower()

    def similarity(self, X, inst):
        sim = []
        for ind in np.array(self.cd):
            dist = np.sqrt(np.sum(np.square(X[ind]-inst)))
            sim.append((0-dist, ind))
        return sim

    def distances(self, inst, metric='euclidean'):
        dist_lst = []
        ind = 0
        dist = 0
        for neighbor in self.cd:
            if metric == 'euclidean':
                dist = np.sqrt(np.sum(np.square(neighbor - inst)))
            elif metric == 'cosine':
                dist = 1 - np.dot(neighbor, inst)/(np.sqrt(np.sum(np.square(dist))) *
                                                   np.sqrt(np.sum(np.square(neighbor))))
            elif metric == 'canberra':
                dist = np.sum(np.abs(neighbor - inst)/ (neighbor + inst))
            dist_lst.append((dist, neighbor, self.y_cd[ind]))
            ind += 1
        return dist_lst

    def acceptable(self, classificationRecord_ind, class_inst_counter, counter):
        if classificationRecord_ind:
            cr_correct = classificationRecord_ind['correct']
        else:
            return False

        #class_accuracy = cr_correct / (counter+1)
        correct_class_accuracy_interval = proportion_confint(cr_correct, counter+1)
        class_freq_interval = proportion_confint(class_inst_counter, counter+1)
        #rel_freq = classCounter_ind / (counter+1)
        #if class_accuracy > rel_freq:

        # If the accuracy interval's lower
        # endpoint is greater than the class frequency interval's higher endpoint, then the instance
        # is accepted.
        if correct_class_accuracy_interval[0] > class_freq_interval[1]:
            print('ATrue', correct_class_accuracy_interval, class_freq_interval)
            return True
        else:
            return False

    def dropInstanceFromCD(self, correct_classificationRecord_ind, class_inst_counter, counter):
        # instances are dropped when their accuracy interval's higher endpoint
        # is less than their class frequency interval's lower endpoint.
        correct_class_accuracy_interval = proportion_confint(correct_classificationRecord_ind, counter + 1)
        class_freq_interval = proportion_confint(class_inst_counter, counter + 1)
        if correct_class_accuracy_interval[1] < class_freq_interval[0]:
            # drop
            #print('DTrue', correct_class_accuracy_interval, class_freq_interval)
            return True
        else:
            return False

    def fitIB1(self, X, y):
        self.cd = [0]
        for ind in range(1, X.shape[0]):
            sim = self.similarity(X, X[ind])
            sim_max = max(sim)

            #print(X[ind], y[ind])
            #print(X[sim_max[1]], y[sim_max[1]])
            #print()
            if y[ind] == y[sim_max[1]]:
                self.classification['correct'] += 1
            else:
                self.classification['incorrect'] += 1
                self.misclassified.append(ind)

            self.cd.append(ind)
        self.y_cd = y[self.cd]
        self.labels_freq = Counter(self.y_cd)
        # convert the indexes of X to data points to use in predict func
        self.cd = X[self.cd]
        #print(self.classification)
        #print(X[self.misclassified])
        #print(y[self.misclassified])

    def fitIB2(self, X, y):
        self.cd = [0]
        for ind in range(1, X.shape[0]):
            sim = self.similarity(X, X[ind])
            sim_max = max(sim)

            #print(X[ind], y[ind])
            #print(X[sim_max[1]], y[sim_max[1]])
            #print()
            if y[ind] == y[sim_max[1]]:
                self.classification['correct'] += 1
            else:
                self.classification['incorrect'] += 1
                self.misclassified.append(ind)

                self.cd.append(ind)
        self.y_cd = y[self.cd]
        self.labels_freq = Counter(self.y_cd)
        # convert the indexes of X to data points to use in predict func
        self.cd = X[self.cd]
        #print(self.cd)
        #print(self.classification)
        #print(X[self.misclassified])
        #print(y[self.misclassified])

    def fitIB3(self, X, y):
        self.cd = [0]
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

            # sim => (max sim, corresponding saved instance in self.cd)
            sim = self.similarity(X, X[ind])

            sim_acceptable = []
            for cd_elem in self.cd:
                class_rec_cdInd = classificationRecord.get(cd_elem)
                #class_coutner_total = np.count_nonzero(y == y[self.cd_elem]
                class_inst_counter = classCounter[y[cd_elem]]
                if self.acceptable(class_rec_cdInd,
                    class_inst_counter,
                    ind):
                    sim_acceptable.append(sim[self.cd.index(cd_elem)])
            if sim_acceptable:
                sim_max_tup = max(sim_acceptable)
            else:
                i = randint(0, len(self.cd)-1)
                sim_max_tup = sim[i]

            sim_max = sim_max_tup[0]
            self.cdInst_max = sim_max_tup[1]
            if y[ind] == y[self.cdInst_max]:
                    self.classification['correct'] += 1
            else:
                self.classification['incorrect'] += 1
                self.misclassified.append(ind)
                self.cd.append(ind)
                sim = self.similarity(X, X[ind])

            cd_copy = self.cd.copy()
            sim_copy = sim.copy()
            for cd_ind in range(len(self.cd)):

                if sim[cd_ind][0] >= sim_max:

                    saved_cd = self.cd[cd_ind]
                    if not classificationRecord.get(saved_cd):
                        classificationRecord[saved_cd] = {'correct': 0,
                                                          'incorrect': 0}
                    if y[ind] == y[saved_cd]:
                        classificationRecord[saved_cd]['correct'] += 1
                    else:
                        classificationRecord[saved_cd]['incorrect'] += 1

                    '''
                    if ((classificationRecord[saved_self.cd]['incorrect'] > th_lower or
                        classificationRecord[saved_self.cd]['correct'] < th_upper) and
                        len(self.cd_copy)>1):
                    '''
                    class_rec_cdInd = classificationRecord[saved_cd]['correct']
                    # class_coutner_total = np.count_nonzero(y == y[self.cd_elem]
                    class_inst_counter = classCounter[y[saved_cd]]
                    if self.dropInstanceFromCD(class_rec_cdInd,
                        class_inst_counter,
                        ind):
                        cd_copy.remove(saved_cd)
                        sim_copy.remove(sim[cd_ind])
            self.cd = cd_copy
        self.y_cd = y[self.cd]
        self.labels_freq = Counter(self.y_cd)
        # convert the indexes of X to data points to use in predict func
        self.cd = X[self.cd]

    def fit(self, dataX, datay):
        if isinstance(dataX, pd.DataFrame) or isinstance(dataX, pd.core.series.Series):
            X = dataX.values
        elif isinstance(dataX, np.ndarray):
            X = dataX
        else:
            raise Exception('dataX should be a DataFrame or a numpy array')

        if isinstance(datay, pd.DataFrame) or isinstance(datay, pd.Series):
            y = datay.values
        elif isinstance(datay, np.ndarray):
            y = datay
        else:
            raise Exception('datay should be a DataFrame or a numpy array')

        self.classification = {'correct': 0,
                               'incorrect': 0}
        self.classificationTest = {'correct': 0,
                                   'incorrect': 0}
        self.misclassified = []
        self.misclassifiedTest = []

        if self.ibl_algo == 'ib1':
            self.fitIB1(X, y)
        elif self.ibl_algo  == 'ib2':
            self.fitIB2(X, y)
        elif self.ibl_algo == 'ib3':
            self.fitIB3(X, y)

    def tieResolver(self, most_commons):
        pred_labels, _ = zip(*most_commons)
        freqs = []
        N = 0
        for label in pred_labels:
            freq = self.labels_freq[label]
            N += freq
            freqs.append(freq)
        prob = np.array(freqs) / N

        return np.random.choice(pred_labels, 1, p=prob)[0]

    def getMostCommonLabels(self, cnt_lst):
        cnt = Counter(cnt_lst)
        most_common_num = cnt.most_common(1)[0][1]
        # Check if there are other labels with the same count
        most_commons = list(filter(
            lambda t: t[1] >= most_common_num, cnt.most_common()))
        return most_commons

    def getWinnerLabel(self, neighborhood, voting='mvs'):
        _, Xneighborhood, train_label = zip(*neighborhood)
        pred_label = None
        if voting == 'mvs':
            most_commons = self.getMostCommonLabels(train_label)
            if len(most_commons) == 1:
                pred_label = most_commons[0][0]
            else:
                # tie
                pred_label = self.tieResolver(most_commons)

        elif voting == 'mp':
            most_commons = self.getMostCommonLabels(train_label)
            if len(most_commons) == 1:
                pred_label = most_commons[0][0]
            else:
                # tie
                while pred_label is None:
                    # remove the one which farther
                    train_label = train_label[:-1]
                    most_commons = self.getMostCommonLabels(train_label)
                    if len(most_commons) == 1:
                        pred_label = most_commons[0][0]

        elif voting == 'brd':
            cnt = Counter(train_label)
            borda_counter = 1
            for lbl in train_label:
                cnt[lbl] += len(train_label) - borda_counter
                borda_counter += 1
            most_common_num = cnt.most_common(1)[0][1]
            # Check if there are other labels with the same count
            most_commons = list(filter(
                lambda t: t[1] >= most_common_num, cnt.most_common()))
            if len(most_commons) == 1:
                pred_label = most_commons[0][0]
            else:
                # tie
                pred_label = self.tieResolver(most_commons)

        return pred_label


    def predict(self, dataX, datay, voting='mvs', distance_metric='euclidean'):
        if isinstance(dataX, pd.DataFrame) or isinstance(dataX, pd.Series):
            X = dataX.values
        elif isinstance(dataX, np.ndarray):
            X = dataX
        else:
            raise Exception('dataX should be a DataFrame or a numpy array')

        if isinstance(datay, pd.DataFrame) or isinstance(datay, pd.Series):
            y = datay.values
        elif isinstance(datay, np.ndarray):
            y = datay
        else:
            raise Exception('datay should be a DataFrame or a numpy array')

        for inst, true_label in zip(X, y):
            dist = self.distances(inst, metric=distance_metric)
            neighborhood = []
            for k in range(self.n_neighbors):
                # we don't need to sort all similarities just take the
                # k max similarity neighbors
                min_inst = min(dist)
                dist.remove(min_inst)
                neighborhood.append(min_inst)
            pred = self.getWinnerLabel(neighborhood, voting)

            # feedback to training
            if pred == true_label:
                self.classificationTest['correct'] += 1
            else:
                self.classificationTest['incorrect'] += 1
                self.misclassifiedTest.append(inst)
                self.cd = np.vstack([self.cd, inst])
                self.y_cd = np.append(self.y_cd, true_label)
            #results = [i[1] for i in sorted(neighborhood)[:self.n_neighbors]]
            #print(results)
            #result = Counter(results[0][1].tolist()).most_common(1)

        #print(42,result)
'''
data = np.array([[11,13],
                 [2,3],
                 [3,5],
                 [10,12],
                 [12,10],
                 [1,4]])

y = np.array([1,1,1,2,2,2])

neigh = MyIBL(1, 'ib2')
neigh.fit(data, y)
neigh.predict(np.array([[4,6],[7,8]]))
'''
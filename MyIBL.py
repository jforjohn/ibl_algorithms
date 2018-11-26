import numpy as np
import pandas as pd


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

		#print(cd)
		#print(self.classification)
		#print(X[self.misclassified])
		#print(y[self.misclassified])

	def fitIB3(self, X, y):
		cd = [0]
		classCounter = {}
		classCounter[y[0]] = 1
		classificationRecord = {}
		th_lower = 0.75
		th_upper = 0.90

		for ind in range(1, X.shape[0]):
			# sim => (max sim, correspoding saved instance in cd)
			sim = self.similarity(X, X[ind], np.array(cd))

			'''
			for cd_elem in cd:
				acceptable(classificationRecord.get(cd_elem)['correct'], 
					->classCounter.get(y[cd_elem]) or np.count_nonzero(y == y[cd_elem]),
					ind)
			'''
			sim_max_tup = max(sim)

			sim_max = sim_max_tup[0]
			cdInst_max = sim_max_tup[1]

			if y[ind] == y[cdInst_max]:
					self.classification['correct'] += 1
			else:
				self.classification['incorrect'] += 1
				self.misclassified.append(ind)
				cd.append(ind)
				if not classCounter.get(y[ind]):
					classCounter[y[ind]] = 1
				else:
					classCounter[y[ind]] += 1 

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

					#if (classificationRecord[saved_cd]['incorrect']/ind > th_lower or 
					#	classificationRecord[saved_cd]['correct']/ind < th_upper):
					'''
					if acceptable(classificationRecord.get(cd_elem)['correct'], 
						->classCounter.get(y[cd_elem]) or np.count_nonzero(y == y[cd_elem]),
						ind)
						cd.remove(saved_cd)
					'''
		
		def acceptable(classificationRecord_ind, classCounter_ind, counter):
			class_accuracy = classificationRecord_ind / counter
			rel_freq = classCounter_ind / counter
			if class_accuracy > rel_freq:
				return True
			else:
				return False


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

data = np.array([[2,3],
                 [3,5],
                 [1,4],
                 [10,12],
                 [11,13],
                 [12,10]])

y = np.array([1,1,1,2,2,2])

neigh = MyIBL(1, 'ib3')
neigh.fit(data, y)

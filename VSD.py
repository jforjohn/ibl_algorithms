
##
import pandas as pd
import numpy as np


X_train = pd.DataFrame({'a': ['s', 't', 's', 't','s','t','s'],
                        'b': [10, 35, 12, 40, 11, 42, 15],
                        'c': ['j','k','l','j','l','j','k']})

y_train = pd.DataFrame({'target_var': ['y', 'n', 'y', 'n', 'y', 'n', 'y']})


##

# #Reading file
# url="https://raw.githubusercontent.com/rajsiddarth/Datasets/master/Bank_dataset.csv"
# #import pandas as pd
# data=pd.read_csv(url,header=0,names=["id","age","experience","income","zipcode","family"
#              ,"ccavg","education","mortgage","pers_loan","sec_amount","cd_account","online","credit_card"])
#
# #Removing id,zipcode and experience
# data.drop(['id','experience','zipcode'],inplace=True,axis=1)
# categ_data=data.loc[:,['family','education','pers_loan','sec_amount','cd_account','online','credit_card']]
# data.drop(['family','education','pers_loan','sec_amount','cd_account','online','credit_card'],inplace=True,axis=1)
#
# # Converting to categorical variables
# for i in categ_data.columns:
#     categ_data[i] = categ_data[i].astype('category')
#
# # Using mean normalization for numerical variables
# from sklearn import preprocessing
#
# min_max_scaler = preprocessing.MinMaxScaler()
# data = pd.DataFrame(min_max_scaler.fit_transform(data.values), columns=['age', 'income', 'ccavg', 'mortgage'])
# data = pd.concat([data, categ_data], axis=1)
#
#
# #Separating  evaluation set using stratified sampling
# import random
# random.seed(123)
# from sklearn.cross_validation import StratifiedShuffleSplit,StratifiedKFold
# X,Y=data.ix[:,:-1],data.ix[:,-1]
# data_index=StratifiedShuffleSplit(Y,n_iter=1,test_size=0.3)
# for train_index,test_index in data_index:
#     X_data,X_evaldata=X.loc[train_index],X.loc[test_index]
#     Y_data,Y_evaldata=Y.loc[train_index],Y.loc[test_index]

# Concatenate the data and test set with target
data=pd.concat([X_train,y_train],axis=1).reset_index(drop=True)
#test_data=pd.concat([X_test,y_test],axis=1).reset_index(drop=True)



# #Checking the sampling for data and eval data
# print(data['pers_loan'].value_counts())
# print(eval_data['pers_loan'].value_counts())

#Keep eval data separate to test later
import warnings
warnings.filterwarnings("ignore")

##
#num_attributes=['age','income','ccavg','mortgage']
num_attributes = data.select_dtypes(exclude='object').columns.tolist()
cat_attributes=[item for item in data.columns if item not in num_attributes]
#cat_attributes.pop()
cat_attributes.remove(y_train.columns[0])

##
# Creating empty data frame for probabilities

def vdf_prob_table(dataframe, target_variable):
    prob_table = pd.DataFrame(
        columns=['category', 'category_value', target_variable, 'P' + str('(target_variable/category)')])
    count = 0  # Inititaizing index for probability table

    for categ in cat_attributes:  # Iterating over each category

        for uniq_categ in dataframe[categ].unique():
            temp = dataframe[dataframe[categ] == uniq_categ]
            len_total = len(temp.index)

            for j in dataframe[target_variable].unique():
                temp2 = dataframe[dataframe[target_variable] == j]
                temp2 = temp2[temp2[categ] == uniq_categ]
                len_prob = len(temp2.index)
                prob = len_prob / len_total
                prob_table.loc[
                    count, ['category', 'category_value', target_variable, 'P' + str('(target_variable/category)')]] = [
                    categ, uniq_categ, j, prob]
                count = count + 1

    return prob_table


from sklearn.metrics import euclidean_distances


# Defining euclidean distances
def numeric_distance(dataframe):
    y = euclidean_distances(dataframe[num_attributes], dataframe[num_attributes])

    return y

##
prob_table=vdf_prob_table(data,y_train.columns[0])
prob_table.ix[:,-1]=prob_table.ix[:,-1].astype(float)
euc_distance=numeric_distance(data)
prob_table

##

# Function for calculating value difference measure distances
#import numpy as np


def vdf_distance(dataframe, targetvariable):
    prob_distance = np.zeros(shape=(len(dataframe), len(dataframe)))
    for i in range(0, len(dataframe)):
        temp1 = dataframe.ix[i, :]
        for categ in cat_attributes:
            temp_array = np.ndarray(shape=(len(dataframe), len(dataframe)))
            temp2 = prob_table[(prob_table['category'] == categ) & (prob_table['category_value'] == temp1[categ])]
            temp_target_0 = np.array(temp2[temp2[targetvariable] == 0].ix[:, -1])
            temp_target_1 = np.array(temp2[temp2[targetvariable] == 1].ix[:, -1])
            for k in range(0, len(dataframe)):
                if k != i:
                    temp3 = dataframe.ix[k, :]
                    temp4 = prob_table[
                        (prob_table['category'] == categ) & (prob_table['category_value'] == temp3[categ])]
                    temp_target_01 = np.array(temp4[temp4[targetvariable] == 0].ix[:, -1])
                    temp_target_11 = np.array(temp4[temp4[targetvariable] == 1].ix[:, -1])
                    temp_array[i][k] = abs(temp_target_01 - temp_target_0) + abs(temp_target_1 - temp_target_11)

                else:
                    temp_array[i][k] = 0
            prob_distance = prob_distance + temp_array
    return prob_distance

##

#Calculate total distance
prob_distance=vdf_distance(data, 'target_var')
total_distancematrix=euc_distance+prob_distance


##
#Calculating min distance
min_distanceindex=np.where(total_distancematrix==np.min(total_distancematrix[np.nonzero(total_distancematrix)]))[0]
row=data.ix[min_distanceindex,['pers_loan']]


#The min distance is calculated using custom distance calculations such as euclidean and value difference measur.
#This distance can be used to implement knn algorithm simiar to what is done in default
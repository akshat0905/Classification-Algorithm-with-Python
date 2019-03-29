# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 20:22:09 2019

@author: Admin
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
data = pd.read_csv("C:\\Users\\Admin\\Documents\\Projects_Python\\XYZCorp_LendingData.txt", 
                   encoding = 'utf-8', sep = '\t', low_memory=False)


# Finding the shape of the given data
data.shape

#Finding out columns and their respective data types
col = list(data.columns)
print(col)

n=list(data.dtypes)

print(n)

r={'col_name':col,'dtype':n}

rs=pd.DataFrame(r)

rs

data['grade']=str(data['grade'])



#Checking for nulls


for c in col:
    if (len(data[c][data[c].isnull()])) > 0:
        print("WARNING: Column '{}' has NULL values".format(c))
        


#Deleting columns based on unavailability of values
        
cnp=data.isnull().sum(axis=0)

print(cnp)

row=len(data)


cnp=cnp[cnp > int(row*0.65)]

cnp=list(cnp.index)

data=data.drop(cnp,axis=1)

data.shape

#deleting zip code as it does not provide a useful insight as only first three digits are given
#id as it is unique
#member_id as it is unique as well



dl=['zip_code','id','member_id']

#factor variables

factor_x = data.select_dtypes(exclude=["int64","float64","category"]).columns.values
print(factor_x)

len(data[factor_x[0]].unique())

for c in factor_x:
    print("Factor variable = '" + c + "'")
    print(len(data[c].unique()))
    print("***")

m=[]

for c in factor_x:
    print("Factor variable = '" + c + "'")
    print(len(data[c].unique()))


#To be done
data['earliest_cr_line'].head()
mnth=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


data=data.drop(['earliest_cr_line'],axis=1)
#to be done

#Numeric variables

factor_y = data.select_dtypes(exclude=["category","object"]).columns.values
print(factor_y)

len(factor_y)

data['policy_code']=(data['policy_code'])

y=0
for c in factor_y:
    if (len(data[c][data[c] == 0] > 0)):
        print(c,len(data[c][data[c] == 0]))
        y=y+1
        
        print("WARNING: Column '{}' has value = 0".format(c))

print(factor_y)


#we will be droping policy_code as this is a case of 0 not present.
 
dl.append('policy_code')



data.shape

cols = list(data.columns)
for c in cols:
    if (len(data[c][data[c].isnull()])) > 0:
        print("WARNING: Column '{}' has NULL values".format(c))


len(factor_x)
len(factor_y)

data.shape

#emp_title and title to be deleted because of too many factors
#title
dl.append('emp_title')
dl.append('title')

type(dl)
del dl[-1]
data=data.drop(dl,axis=1)


data.shape

data['revol_util'].head(5)
data['mths_since_last_delinq'].head(5)
data['last_pymnt_d'].head(5)
data['next_pymnt_d'].head(5)
data['last_credit_pull_d'].head(5)
data['collections_12_mths_ex_med'].head(5)
data['tot_coll_amt'].head(5)
data['tot_cur_bal'].head(-5)
data['total_rev_hi_lim'].head(-5)

#revol_util to be treated
#mths_since_last_delinq to be treated, deleting right now as 50% of the data is not present, and it might skew the data if filled with same value
#last_pymnt_d is date value
#Next_pymnt_d is date value delete
#last_credit_pull_d is another date value
#collection_12mths_ex_md should be deleted
#tot_coll_amt it comes after a person becomes a defaulter
#total_rev_hi_lim 
dl.append('last_pymnt_d')
dl.append('next_pymnt_d')
dl.append('last_credit_pull_d')
dl.append('mths_since_last_delinq')

data=data.drop(dl,axis=1)

data.shape



# checking for bias in data
# singular values in factors

data['pymnt_plan'].value_counts(normalize=True) * 100
data['application_type'].value_counts(normalize=True) * 100
data['term'].value_counts(normalize=True) * 100


dl.append('application_type')
dl.append('pymnt_plan')

data=data.drop('application_type',axis=1)
data=data.drop('pymnt_plan',axis=1)
#application_type
#pymnt_plan




import statistics as s

n1=s.median(data['revol_util'])
n2=s.median(data['collections_12_mths_ex_med'])
n3=s.median(data['tot_coll_amt'])
n4=s.median(data['tot_cur_bal'])
n5=s.median(data['total_rev_hi_lim'])


data['revol_util'].fillna(n1, inplace = True)
data['collections_12_mths_ex_med'].fillna(n2, inplace = True)
data['tot_coll_amt'].fillna(n3, inplace = True)
data['tot_cur_bal'].fillna(n4, inplace = True)
data['total_rev_hi_lim'].fillna(n5, inplace = True)



#Creating dummies
factor_x=np.delete(factor_x,7)
for var in factor_x:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data = data1
    
data.shape
#splitting the dataset

data['issue_d']=date['issue_d']

#Deleting the old columns
data=data.drop(factor_x,axis=1)

X = list(train_x.columns)
Y = list(train_y.columns)

#
# reordering the columns
# ---------------------------------------------------
data = pd.concat(
        [data['default_ind'], 
        data.drop('default_ind',axis=1)],
        axis=1)
# converting the columns to ndarray
# ---------------------------------
final_cols = data.columns.values
type(final_cols)
print(final_cols)
len(final_cols)

# converting the columns to list
# ------------------------------
data_final_vars = data.columns.values.tolist()
type(data_final_vars)
Y=['default_ind']
X=[i for i in data_final_vars if i not in y]

print(X)
print(Y)
train.shape
test.shape


#splitting
data=data.drop(['issue_d'],axis=1)


tst_mnth=['Jun-2015','Jul-2015','Aug-2015','Sep-2015','Oct-2015','Nov-2015','Dec-2015']    

train=data.loc[ -data.issue_d.isin(tst_mnth)]
test=data.loc[ data.issue_d.isin(tst_mnth)]

train=train.drop('issue_d',axis=1)
test=test.drop('issue_d',axis=1)
train.shape
test.shape
data.shape

# split the train and test into X and Y variables
train_x = train.iloc[:,0:161]; train_y = train.iloc[:,161]
test_x  = test.iloc[:,0:161];  test_y = test.iloc[:,161]

#####################




from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as cr
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix


# random_state --> tells if the same data be used each time or different
# ---------------------------------------------------
logreg = LogisticRegression(random_state=0)
logreg.fit(train_x, train_y)

# predict on the test set
# ---------------------------------------------------
pred_y = logreg.predict(test_x)
# cf = confusion_matrix(test_y, pred_y, labels=['actual','predicted'])
labels=[0,1]
cf = confusion_matrix(pred_y,test_y,labels)
print(cf)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(test_x, test_y)))

# confusion matrix with details
# -----------------------------
ty=list(test_y)
py=list(pred_y)
cm1=ConfusionMatrix(py,ty)
print(cm1)
cm1.print_stats()
cm1.plot()

# Classification report : precision, recall, F-score
# ---------------------------------------------------
print(cr(test_y, pred_y))


#model number 2

# RFE (recursive feature elimination)
# -----------------------------------
logreg = LogisticRegression()

# sklearn.feature_selection.RFE
# (estimator, n_features_to_select=None, step=1, verbose=0)
# get the best 18 features
rfe = RFE(logreg, 150)
rfe = rfe.fit(data[X], data[Y] )
support = rfe.support_
ranking = rfe.ranking_

len(support); print(support)
len(ranking); type(ranking)

df_rfe = pd.DataFrame({"columns":final_cols[0:161], 
                       "support":support, 
                       "ranking":ranking})
df_rfe.sort_values("ranking")
# as per the RFE, the following columns are the best
# cols = df_rfe[['columns']][(df_rfe.support == True)]

cols = df_rfe['columns'][(df_rfe.support == True)].tolist()
print(cols)
type(cols)
len(cols)


X=data[cols] # train_x
Y=data['default_ind'] # train_y
X.columns

# concat the 2 datasets with the signifcant columns only
# -----------------------------------------------------
new_dataset = pd.concat([X,Y], axis=1)
new_dataset['issue_d']=date['issue_d']

tst_mnth=['Jun-2015','Jul-2015','Aug-2015','Sep-2015','Oct-2015','Nov-2015','Dec-2015']    

train1 = pd.concat(
        [test1['default_ind'], 
        train1.drop('default_ind',axis=1)],
        axis=1)

test1 = pd.concat(
        [test1['default_ind'], 
        test1.drop('default_ind',axis=1)],
        axis=1)



train1=new_dataset.loc[ -new_dataset.issue_d.isin(tst_mnth)]
test1=new_dataset.loc[ new_dataset.issue_d.isin(tst_mnth)]

train1=train1.drop('issue_d',axis=1)
test1=test1.drop('issue_d',axis=1)


train1.shape
test1.shape

# split the train and test into X and Y variables
train_x1 = train1.iloc[:,1:151]; train_y1 = train1.iloc[:,0]
test_x1  = test1.iloc[:,0:150];  test_y1 = test1.iloc[:,150]

train1.shape
test1.shape

train_x1['default_ind']
test_y1['defaul_ind']

logreg = LogisticRegression(random_state=0)
logreg.fit(train_x1, train_y1)

# predict on the test set
# ---------------------------------------------------
pred_y = logreg.predict(test_x1)
# cf = confusion_matrix(test_y, pred_y, labels=['actual','predicted'])
labels=[0,1]
cf = confusion_matrix(pred_y,test_y,labels)
print(cf)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(test_x, test_y)))

# confusion matrix with details
# -----------------------------
ty=list(test_y)
py=list(pred_y)
cm1=ConfusionMatrix(py,ty)
print(cm1)
cm1.print_stats()
cm1.plot()

# Classification report : precision, recall, F-score
# ---------------------------------------------------
print(cr(test_y, pred_y))


from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.datasets import make_imbalance
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from numpy import loadtxt
from pandas import read_csv
from sklearn import datasets
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.metrics import classification_report,make_scorer,accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from statistics import mean, stdev
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyodbc 
import re

#!/usr/bin/env python
# coding: utf-8
nltk.download('punkt')

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=SERVER;'
                      'Database=DB;'
                      'user =USER;'
                      'Trusted_Connection=yes;')

sql_query = pd.read_sql_query('SELECT * FROM SCHEMA.NLPFORMS_PROSPECTIVE',conn)
df = pd.DataFrame(sql_query, columns=['FINDINGS_VALUE','MS_SEVERITY_OUTCOMES'])
# removal of numbers from text
df['FINDINGS_VALUE'] =  df['FINDINGS_VALUE'].apply(lambda x: re.sub(r'\[[0-9]*\]',' ', str(x)))
##removal of extra spaces from the text
df['FINDINGS_VALUE'] =  df['FINDINGS_VALUE'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
##removal of non words characters with space
df['FINDINGS_VALUE'] =  df['FINDINGS_VALUE'].apply(lambda x: re.sub(r'\W',' ', str(x)))
##
df['FINDINGS_VALUE'] =  df['FINDINGS_VALUE'].apply(lambda x: re.sub(r'\d',' ', str(x)))
##
df['FINDINGS_VALUE'] =  df['FINDINGS_VALUE'].apply(lambda x: re.sub(r'\s+',' ', str(x)))


# # tokenization and stopword removal
stop_words = set(stopwords.words('english'))  

def remove_stopwords(sentence):
    word_tokens = word_tokenize(sentence)  
    clean_tokens = [w for w in word_tokens if not w in stop_words]  
    
    return clean_tokens
    
df['FINDINGS_VALUE'] = df['FINDINGS_VALUE'].apply(remove_stopwords)
X=df['FINDINGS_VALUE']
y=df['MS_SEVERITY_OUTCOMES']

# removal of numbers from text
df['FINDINGS_VALUE'] =  df['FINDINGS_VALUE'].apply(lambda x: re.sub(r'\[[0-9]*\]',' ', str(x)))
##removal of extra spaces from the text
df['FINDINGS_VALUE'] =  df['FINDINGS_VALUE'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
##removal of non words characters with space
df['FINDINGS_VALUE'] =  df['FINDINGS_VALUE'].apply(lambda x: re.sub(r'\W',' ', str(x)))
##
df['FINDINGS_VALUE'] =  df['FINDINGS_VALUE'].apply(lambda x: re.sub(r'\d',' ', str(x)))
##
df['FINDINGS_VALUE'] =  df['FINDINGS_VALUE'].apply(lambda x: re.sub(r'\s+',' ', str(x)))

# # VECTORIZATION USING BOW +TF-IDF
count_vect = CountVectorizer(min_df=1,max_df=1.0)

X_counts = count_vect.fit_transform(X)
X_counts.shape
tfidf_transformer = TfidfTransformer()
X_tfidf=tfidf_transformer.fit_transform(X_counts)

X_tfidf
y_new= LabelEncoder().fit_transform(y)
y_new.shape


# # SYNTHETIC MINORITY OVERSAMPLING TECHNIQUE (SMOTE)
oversample = SMOTE()
X_tfidf, y_new = oversample.fit_resample(X_tfidf, y_new)
counterx = Counter(y_new)
for k,v in counterx.items():
    per = v / len(y_new) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

pyplot.bar(counterx.keys(), counterx.values())
pyplot.show()
y_new.shape
X_tfidf.shape

# # RANDOM SAMPLING 7:3 SPLIT 
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_new, test_size=0.33, random_state=42)


# # RANDOM FOREST+(BOW+TFIDF)+SMOTE+TRAIN TEST 7:3
rf= RandomForestClassifier()
text_clf=make_pipeline(rf)
# Train the model on training data
text_clf.fit(X_train,y_train)
# Use the forest's predict method on the test data
predictions = text_clf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=I","2=Mi","3=MM","4=Mo",
                      "5=MS","6=S"])
metrics.accuracy_score(y_test,predictions)


# # SUPPORT VECTOR MACHINES (SVM)+(BOW+TFIDF)+SMOTE+TRAIN TEST 7:3
text_clf=make_pipeline(LinearSVC())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=I","2=Mi","3=MM","4=Mo",
                      "5=MS","6=S"])
print(classification_report(y_test,predictions))


# # XGBOOST+(BOW+TFIDF)+SMOTE+TRAIN TEST 7:3
text_clf=make_pipeline(XGBClassifier())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=I","2=Mi","3=MM","4=Mo",
                      "5=MS","6=S"])
metrics.accuracy_score(y_test,predictions)
print(classification_report(y_test,predictions))
# # 5 FOLD STRATIFIED CROSS VALIDATION RANDOM FOREST+SMOTE+(BOW+TFIDF)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
    score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, Z):
    print (classification_report(y_new, Z)) # print classification report
    return accuracy_score(y_new, Z) # return accuracy score

# Nested CV with parameter optimization
nested_score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)


# # 5 FOLD STRATIFIED CROSS VALIDATION SVM+SMOTE+(BOW+TFIDF)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1
# split()  method generate indices to split data into training and test set.
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
    score = cross_val_score(make_pipeline(LinearSVC()), X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, U):
    print (classification_report(y_new, U)) # print classification report
    return accuracy_score(y_new, U)

# Nested CV with parameter optimization
nested_score = cross_val_score(make_pipeline(LinearSVC()), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
# # 5 FOLD STRATIFIED CROSS VALIDATION XGBOOST+SMOTE+(BOW+TFIDF)
get_ipython().system('pip install xgboost')

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
score = cross_val_score(make_pipeline(XGBClassifier()), X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, V):
    print (classification_report(y_new, V)) # print classification report
    return accuracy_score(y_new, V)

# Nested CV with parameter optimization
nested_score = cross_val_score(make_pipeline(XGBClassifier()), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)


# # SMOTE+ENN+TFIDF+BOW
oversample = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='all'))
X_tfidf, y_new = oversample.fit_resample(X_tfidf, y_new)
counterx = Counter(y_new)
for k,v in counterx.items():
    per = v / len(y_new) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

pyplot.bar(counterx.keys(), counterx.values())
pyplot.show()


# # SMOTE+ENN+TFIDF+BOW+TRAIN TEST SPLIT (7:3)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_new, test_size=0.33, random_state=42)


# # SMOTE+ENN+TFIDF+BOW+TRAIN TEST SPLIT (7:3)+random forest
rf= RandomForestClassifier()
text_clf=make_pipeline(rf)
# Train the model on training data
text_clf.fit(X_train,y_train)
# Use the forest's predict method on the test data
predictions = text_clf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=I","2=Mi","3=MM","4=Mo",
                      "5=MS","6=S"])
metrics.accuracy_score(y_test,predictions)
# # SMOTE+ENN+TFIDF+BOW+TRAIN TEST SPLIT (7:3)+SVM

text_clf=make_pipeline(LinearSVC())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)

print(confusion_matrix(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=I","2=Mi","3=MM","4=Mo",
                      "5=MS","6=S"])
print(classification_report(y_test,predictions))


# # SMOTE+ENN+TFIDF+BOW+TRAIN TEST SPLIT (7:3)+XGBOOST
text_clf=make_pipeline(XGBClassifier())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=I","2=Mi","3=MM","4=Mo",
                      "5=MS","6=S"])
metrics.accuracy_score(y_test,predictions)
print(classification_report(y_test,predictions))
# # STRATIFIED 5 FOLD CV+SMOTE +ENN +TFIDF+BOW+ RANDOM FOREST
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1

score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, Z):
    print (classification_report(y_new, Z)) # print classification report
    return accuracy_score(y_new, Z) # return accuracy score

# Nested CV with parameter optimization
nested_score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)


# # STRATIFIED 5 FOLD CV+SMOTE +ENN +TFIDF+BOW+ SVM
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1

score = cross_val_score(make_pipeline(LinearSVC()), X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, U):
    print (classification_report(y_new, U)) # print classification report
    return accuracy_score(y_new, U)

# Nested CV with parameter optimization
nested_score = cross_val_score(make_pipeline(LinearSVC()), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)


# # STRATIFIED 5 FOLD CV+SMOTE +ENN +TFIDF+BOW+ XGBoost

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1

score = cross_val_score(make_pipeline(XGBClassifier()), X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, d):
    print (classification_report(y_new, d)) # print classification report
    return accuracy_score(y_new, d)

# Nested CV with parameter optimization
nested_score = cross_val_score(make_pipeline(XGBClassifier()), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
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

nltk.download('punkt')
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=SERVER;'
                      'Database=DB;'
                      'user =USER;'
                      'Trusted_Connection=yes;')
sql_query = pd.read_sql_query('SELECT * FROM SCHEMA.NLPFORDF_PROSPECTIVE',conn)
df = pd.DataFrame(sql_query, columns=['Findings_Value','DF_SEVERITY_OUTCOMES'])
print(df)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\[[0-9]*\]',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\W',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\d',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+[a-z]\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'^[a-z]\s+',' ', str(x)))
stop_words = set(stopwords.words('english'))  
def remove_stopwords(sentence):
    word_tokens = word_tokenize(sentence)  
    clean_tokens = [w for w in word_tokens if not w in stop_words]  
    return clean_tokens
df['Findings_Value'] = df['Findings_Value'].apply(remove_stopwords)
X=df['Findings_Value']
y=df['DF_SEVERITY_OUTCOMES']
print(df)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\[[0-9]*\]',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\W',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\d',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
print(df)
count_vect = CountVectorizer(min_df=1,max_df=1.0)
X_counts = count_vect.fit_transform(X)
X_counts.shape
tfidf_transformer = TfidfTransformer()
X_tfidf=tfidf_transformer.fit_transform(X_counts)
X_tfidf
y_new= LabelEncoder().fit_transform(y)
y_new.shape
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
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_new, test_size=0.33, random_state=42)
rf= RandomForestClassifier()
text_clf=make_pipeline(rf)
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
errors = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=NORMAL","2=G1","3=G2","4=G3"])
metrics.accuracy_score(y_test,predictions)
text_clf=make_pipeline(LinearSVC())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=NORMAL","2=G1","3=G2","4=G3"])
print(classification_report(y_test,predictions))
text_clf=make_pipeline(XGBClassifier())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=NORMAL","2=G1","3=G2","4=G3"])
metrics.accuracy_score(y_test,predictions)
print(classification_report(y_test,predictions))
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
nested_score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
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
nested_score = cross_val_score(make_pipeline(LinearSVC()), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
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
nested_score = cross_val_score(make_pipeline(XGBClassifier()), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
oversample = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='all'))
X_tfidf, y_new = oversample.fit_resample(X_tfidf, y_new)
counterx = Counter(y_new)
for k,v in counterx.items():
    per = v / len(y_new) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
pyplot.bar(counterx.keys(), counterx.values())
pyplot.show()
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_new, test_size=0.33, random_state=42)
rf= RandomForestClassifier()
text_clf=make_pipeline(rf)
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
errors = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=NORMAL","2=G1","3=G2","4=G3"])
metrics.accuracy_score(y_test,predictions)
text_clf=make_pipeline(LinearSVC())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=NORMAL","2=G1","3=G2","4=G3"])
print(classification_report(y_test,predictions))
text_clf=make_pipeline(XGBClassifier())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test,display_labels =["0=NA","1=NORMAL","2=G1","3=G2","4=G3"])
metrics.accuracy_score(y_test,predictions)
print(classification_report(y_test,predictions))
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
nested_score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
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
nested_score = cross_val_score(make_pipeline(LinearSVC()), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
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
nested_score = cross_val_score(make_pipeline(XGBClassifier()), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
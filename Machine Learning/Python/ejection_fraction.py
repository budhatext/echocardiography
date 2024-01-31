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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
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

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=server;'
                      'Database=db;'
                      'user =user;'
                      'Trusted_Connection=yes;')
sql_query = pd.read_sql_query('SElect * from SCHEMA.POINT_ESTIMATES_NLP_PROSPECTIVE', conn)
df = pd.DataFrame(sql_query, columns=['Findings_Value','EF_OUTCOMES'])
print(df)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+[a-z]\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'^[a-z]\s+',' ', str(x)))
print(df)
stop_words = set(stopwords.words('english'))  
def remove_stopwords(sentence):
    word_tokens = word_tokenize(sentence)  
    clean_tokens = [w for w in word_tokens if not w in stop_words]  
    return clean_tokens
df['Findings_Value'] = df['Findings_Value'].apply(remove_stopwords)
print(df)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\[[0-9]*\]',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\W',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df.head()
print(df)
df.isnull().sum()
df['EF_OUTCOMES'].value_counts()
X=df['Findings_Value']
y=df['EF_OUTCOMES']
count_vect = CountVectorizer(min_df=1,max_df=1.0)
X_counts = count_vect.fit_transform(X)
X_counts.shape
print(X_counts)
tfidf_transformer = TfidfTransformer()
X_tfidf=tfidf_transformer.fit_transform(X_counts)
X_tfidf
y_new= LabelEncoder().fit_transform(y)
y_new.shape
X_tfidf.shape
oversample = SMOTE()
X_tfidf, y_new = oversample.fit_resample(X_tfidf, y_new)
counterx = Counter(y_new)
for k,v in counterx.items():
    per = v / len(y_new) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
pyplot.bar(counterx.keys(), counterx.values())
pyplot.show()
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_new,test_size=0.33,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_new,test_size=0.33,random_state=42)
n_estimators_list=list(range(10,220,50))
criterion_list=['gini','entropy']
max_depth_list=list(range(5,41,10))
max_depth_list.append(None)
min_samples_split_list=[x/1000 for x in list (range (5,41,10))]
min_samples_leaf_list=[x/1000 for x in list (range (5,41,10))]
max_features_list=['sqrt','log2']
params_grid= {
    'n_estimators': n_estimators_list,
    'criterion': criterion_list,
    'max_depth': max_depth_list,
    'min_samples_split': min_samples_split_list,
    'min_samples_leaf': min_samples_leaf_list,
    'max_features': max_features_list        
}
num_combinations = 1
for k in params_grid.keys(): num_combinations *=len(params_grid[k])
print('Number of combinations =', num_combinations)
params_grid

def my_roc_auc_score(model, X_train, y_train): return metrics.roc_auc_score(y_test, model.predict(X_test))
model_rf= RandomizedSearchCV(estimator=RandomForestClassifier(),
                             param_distributions=params_grid,
                             n_iter=50,
                             cv=10,
                             return_train_score=True)
model_rf.fit(X_train,y_train)
model_rf.best_params_
model_rf_fin=RandomForestClassifier(
                                   criterion='gini',
                                   max_features='log2',
                                   min_samples_leaf=0.005,
                                   min_samples_split=0.005,
                                   n_estimators=210,
                                   max_depth=35,
                                   random_state=42)
model_rf_fin.fit(X_train,y_train)
predictions = model_rf_fin.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(model_rf_fin,X_test,y_test)
metrics.accuracy_score(y_test,predictions)
text_clf=make_pipeline(LinearSVC())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
text_clf=make_pipeline(XGBClassifier())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
oversample = SMOTE()
X_tfidf, y_new = oversample.fit_resample(X_tfidf, y_new)
counterx = Counter(y_new)
for k,v in counterx.items():
    per = v / len(y_new) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
pyplot.bar(counterx.keys(), counterx.values())
pyplot.show()
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_new,test_size=0.33,random_state=42)
n_estimators_list=list(range(10,220,50))
criterion_list=['gini','entropy']
max_depth_list=list(range(5,41,10))
max_depth_list.append(None)
min_samples_split_list=[x/1000 for x in list (range (5,41,10))]
min_samples_leaf_list=[x/1000 for x in list (range (5,41,10))]
max_features_list=['sqrt','log2']
params_grid= {
    'n_estimators': n_estimators_list,
    'criterion': criterion_list,
    'max_depth': max_depth_list,
    'min_samples_split': min_samples_split_list,
    'min_samples_leaf': min_samples_leaf_list,
    'max_features': max_features_list        
}
num_combinations = 1
for k in params_grid.keys(): num_combinations *=len(params_grid[k])
print('Number of combinations =', num_combinations)
params_grid
def my_roc_auc_score(model, X_train, y_train): return metrics.roc_auc_score(y_test, model.predict(X_test))
model_rf= RandomizedSearchCV(estimator=RandomForestClassifier (class_weight='balanced'),
                             param_distributions=params_grid,
                             n_iter=50,
                             cv=10,
                             return_train_score=True)
model_rf.fit(X_train,y_train)
model_rf.best_params_
model_rf_fin=RandomForestClassifier(
                                   criterion='gini',
                                   max_features='sqrt',
                                   min_samples_leaf=0.005,
                                   min_samples_split=0.015,
                                   n_estimators=210,
                                   max_depth=5,
                                   random_state=42)
model_rf_fin.fit(X_train,y_train)
predictions = model_rf_fin.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(model_rf_fin,X_test,y_test)
metrics.accuracy_score(y_test,predictions)
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
text_clf=make_pipeline(LinearSVC())
score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, Z):
    print (classification_report(y_new, Z)) # print classification report
    return accuracy_score(y_new, Z) # return accuracy score
nested_score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
text_clf=make_pipeline(XGBClassifier())
score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, Z):
    print (classification_report(y_new, Z)) # print classification report
    return accuracy_score(y_new, Z) # return accuracy score
nested_score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=server;'
                      'Database=db;'
                      'user =user;'
                      'Trusted_Connection=yes;')
sql_query = pd.read_sql_query('SElect * from SCHEMA.POINT_ESTIMATES_NLP_PROSPECTIVE', conn)
df = pd.DataFrame(sql_query, columns=['Findings_Value','EF_OUTCOMES'])

df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+[a-z]\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'^[a-z]\s+',' ', str(x)))
stop_words = set(stopwords.words('english'))  
def remove_stopwords(sentence):
    word_tokens = word_tokenize(sentence)  
    clean_tokens = [w for w in word_tokens if not w in stop_words]  
    return clean_tokens
df['Findings_Value'] = df['Findings_Value'].apply(remove_stopwords)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\[[0-9]*\]',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\W',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
X=df['Findings_Value']
y=df['EF_OUTCOMES']
count_vect = CountVectorizer(min_df=1,max_df=1.0)
X_counts = count_vect.fit_transform(X)
tfidf_transformer = TfidfTransformer()
X_tfidf=tfidf_transformer.fit_transform(X_counts)
y_new= LabelEncoder().fit_transform(y)
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
plot_confusion_matrix(text_clf,X_test,y_test)
text_clf=make_pipeline(LinearSVC())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
print(classification_report(y_test,predictions))
text_clf=make_pipeline(XGBClassifier())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
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
nested_score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
text_clf=make_pipeline(LinearSVC())
score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, Z):
    print (classification_report(y_new, Z)) # print classification report
    return accuracy_score(y_new, Z) # return accuracy score
nested_score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
text_clf=make_pipeline(XGBClassifier())
score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, Z):
    print (classification_report(y_new, Z)) # print classification report
    return accuracy_score(y_new, Z) # return accuracy score
nested_score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)

nltk.download('punkt')
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=server;'
                      'Database=db;'
                      'user =user;'
                      'Trusted_Connection=yes;')
sql_query = pd.read_sql_query('SELECT * FROM SCHEMA.EF_BANDS_PROSPECTIVE',conn)

df = pd.DataFrame(sql_query, columns=['Findings_Value','FINAL_EF'])
print(df)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+[a-z]\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'^[a-z]\s+',' ', str(x)))
print(df)
stop_words = set(stopwords.words('english'))  

def remove_stopwords(sentence):
    word_tokens = word_tokenize(sentence)  
    clean_tokens = [w for w in word_tokens if not w in stop_words]  
    return clean_tokens
df['Findings_Value'] = df['Findings_Value'].apply(remove_stopwords)
print(df)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\[[0-9]*\]',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\W',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df.head()
print(df)
df.isnull().sum()
df['FINAL_EF'].value_counts()
X=df['Findings_Value']
y=df['FINAL_EF']

count_vect = CountVectorizer(min_df=1,max_df=1.0)
X_counts = count_vect.fit_transform(X)
X_counts.shape
tfidf_transformer = TfidfTransformer()
X_tfidf=tfidf_transformer.fit_transform(X_counts)
X_tfidf
y_new= LabelEncoder().fit_transform(y)
y_new.shape
X_tfidf.shape
oversample = SMOTE()
X_tfidf, y_new = oversample.fit_resample(X_tfidf, y_new)
counterx = Counter(y_new)
for k,v in counterx.items():
    per = v / len(y_new) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
pyplot.bar(counterx.keys(), counterx.values())
pyplot.show()
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_new,test_size=0.33,random_state=42)
rf= RandomForestClassifier()
print(y_train)
y_new= LabelEncoder().fit_transform(y)
print(y_train)
text_clf=make_pipeline(rf)
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
text_clf=make_pipeline(LinearSVC())
text_clf.fit(X_train,y_train)
y_score=text_clf.fit(X_train,y_train).decision_function(X_test)
print(y_score)
predictions=text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
metrics.accuracy_score(y_test,predictions)
print("Accuracy", metrics.accuracy_score(y_test, predictions))
metrics.accuracy_score(y_test,predictions)
text_clf=make_pipeline(XGBClassifier())
text_clf.fit(X_train,y_train)
predictions=text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
X=df['Findings_Value']
y=df['FINAL_EF']
count_vect = CountVectorizer(min_df=1,max_df=1.0)
X_counts = count_vect.fit_transform(X)
y_new= LabelEncoder().fit_transform(y)
tfidf_transformer = TfidfTransformer()
X_tfidf=tfidf_transformer.fit_transform(X_counts)
oversample = SMOTE()
X_tfidf, y_new = oversample.fit_resample(X_tfidf, y_new)
counterx = Counter(y_new)
for k,v in counterx.items():
    per = v / len(y_new) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
pyplot.bar(counterx.keys(), counterx.values())
pyplot.show()
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
nltk.download('punkt')
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=SERVER;'
                      'Database=DB;'
                      'user =USER;'
                      'Trusted_Connection=yes;')

sql_query = pd.read_sql_query('SELECT Findings_Value,FINAL_EF FROM EF_BANDS_PROSPECTIVE',conn)
df = pd.DataFrame(sql_query, columns=['Findings_Value','FINAL_EF'])
print(df)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+[a-z]\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'^[a-z]\s+',' ', str(x)))
stop_words = set(stopwords.words('english'))  

def remove_stopwords(sentence):
    word_tokens = word_tokenize(sentence)  
    clean_tokens = [w for w in word_tokens if not w in stop_words]  
    return clean_tokens
df['Findings_Value'] = df['Findings_Value'].apply(remove_stopwords)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\[[0-9]*\]',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\W',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
X=df['Findings_Value']
y=df['FINAL_EF']
count_vect = CountVectorizer(min_df=1,max_df=1.0)
X_counts = count_vect.fit_transform(X)
tfidf_transformer = TfidfTransformer()
X_tfidf=tfidf_transformer.fit_transform(X_counts)
y_new= LabelEncoder().fit_transform(y)
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
plot_confusion_matrix(text_clf,X_test,y_test)
text_clf=make_pipeline(LinearSVC())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
print(classification_report(y_test,predictions))
text_clf=make_pipeline(XGBClassifier())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
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
nested_score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
text_clf=make_pipeline(LinearSVC())
score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, Z):
    print (classification_report(y_new, Z)) # print classification report
    return accuracy_score(y_new, Z) # return accuracy score
nested_score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
text_clf=make_pipeline(XGBClassifier())
score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, Z):
    print (classification_report(y_new, Z)) # print classification report
    return accuracy_score(y_new, Z) # return accuracy score
nested_score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)

nltk.download('punkt')
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=SERVER;'
                      'Database=DB;'
                      'user =USER;'
                      'Trusted_Connection=yes;')
sql_query = pd.read_sql_query('SELECT * FROM SCHEMA.TEXTEF_PROSPECTIVE',conn)
df = pd.DataFrame(sql_query, columns=['Findings_Value','FINAL_EF'])
print(df)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+[a-z]\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'^[a-z]\s+',' ', str(x)))
print(df)
stop_words = set(stopwords.words('english'))  

def remove_stopwords(sentence):
    word_tokens = word_tokenize(sentence)  
    clean_tokens = [w for w in word_tokens if not w in stop_words]  
    return clean_tokens
df['Findings_Value'] = df['Findings_Value'].apply(remove_stopwords)
print(df)
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\[[0-9]*\]',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\W',' ', str(x)))
df['Findings_Value'] =  df['Findings_Value'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
df.head()
print(df)
df.isnull().sum()
df['FINAL_EF'].value_counts()
X=df['Findings_Value']
y=df['FINAL_EF']
count_vect = CountVectorizer(min_df=1,max_df=1.0)
X_counts = count_vect.fit_transform(X)
tfidf_transformer = TfidfTransformer()
X_tfidf=tfidf_transformer.fit_transform(X_counts)
X_tfidf
y_new= LabelEncoder().fit_transform(y)
y_new.shape
X_tfidf.shape
oversample = SMOTE()
X_tfidf, y_new = oversample.fit_resample(X_tfidf, y_new)
counterx = Counter(y_new)
for k,v in counterx.items():
    per = v / len(y_new) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
pyplot.bar(counterx.keys(), counterx.values())
pyplot.show()
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_new,test_size=0.33,random_state=42)
rf= RandomForestClassifier()
text_clf=make_pipeline(rf)
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
text_clf=make_pipeline(LinearSVC())
text_clf.fit(X_train,y_train)
y_score=text_clf.fit(X_train,y_train).decision_function(X_test)
print(y_score)
predictions=text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
metrics.accuracy_score(y_test,predictions)
text_clf=make_pipeline(XGBClassifier())
text_clf.fit(X_train,y_train)
predictions=text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
oversample = SMOTE()
X_tfidf, y_new = oversample.fit_resample(X_tfidf, y_new)
counterx = Counter(y_new)
for k,v in counterx.items():
    per = v / len(y_new) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
pyplot.bar(counterx.keys(), counterx.values())
pyplot.show()
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
oversample = SMOTE()
X_tfidf, y_new = oversample.fit_resample(X_tfidf, y_new)
counterx = Counter(y_new)
for k,v in counterx.items():
    per = v / len(y_new) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
pyplot.bar(counterx.keys(), counterx.values())
pyplot.show()
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
text_clf=make_pipeline(LinearSVC())
score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, Z):
    print (classification_report(y_new, Z)) # print classification report
    return accuracy_score(y_new, Z) # return accuracy score
nested_score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cnt = 1
for train_index, test_index in kf.split(X_tfidf, y_new):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1
text_clf=make_pipeline(XGBClassifier())
score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

def classification_report_with_accuracy_score(y_new, Z):
    print (classification_report(y_new, Z)) # print classification report
    return accuracy_score(y_new, Z) # return accuracy score
nested_score = cross_val_score(text_clf, X_tfidf, y_new, cv= kf, \
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
plot_confusion_matrix(text_clf,X_test,y_test)
text_clf=make_pipeline(LinearSVC())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
print(classification_report(y_test,predictions))
text_clf=make_pipeline(XGBClassifier())
text_clf.fit(X_train,y_train)
predictions = text_clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
plot_confusion_matrix(text_clf,X_test,y_test)
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

def classification_report_with_accuracy_score(y_new, V):
    print (classification_report(y_new, V)) # print classification report
    return accuracy_score(y_new, V)
nested_score = cross_val_score(make_pipeline(XGBClassifier()), X_tfidf, y_new, cv= kf, \
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)
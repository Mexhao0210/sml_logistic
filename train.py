import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

# %matplotlib inline
from sklearn.pipeline import Pipeline

df = pd.read_csv('token.csv')
df = df[pd.notnull(df['tags'])]
test=pd.read_csv('token_test.csv')
test=(test.post).values.astype('U')
# print(test)
X = (df.post).values.astype('U')
y = df.tags
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 55)
# print(X_train)
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X, y)

# %%time

y_pred = logreg.predict(test)
with open('result.txt','w',encoding='utf-8') as result:
    for i in y_pred:
        result.write(i)
# print('accuracy %s' % accuracy_score(y_pred, y_test))
# print(classification_report(y_test, y_pred,target_names=my_tags))
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
features_test=pd.read_csv('dataset.csv')
url_list=features_test['url']
y=features_test['label']
def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/') # make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))#remove redundant tokens
    if 'https' in total_Tokens:
        total_Tokens.remove('https')#removing https
    if 'http' in total_Tokens:
        total_Tokens.remove('http')#removing http
    if 'com' in total_Tokens:
        total_Tokens.remove('com')#removing .com 
    return total_Tokens
vectorizer = TfidfVectorizer(tokenizer=makeTokens)
X = vectorizer.fit_transform(url_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

joblib.dump(logreg,'logreg.pkl')
joblib.dump(vectorizer,'vec.pkl')
joblib.dump(X_test,'X_test.pkl')
joblib.dump(y_test,'y_test.pkl')
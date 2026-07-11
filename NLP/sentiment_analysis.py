"""
Sentiment Analysis Project (Simple End-to-End)

NOTE:
This is a starter template with simple code and comments.
Install:
pip install pandas nltk scikit-learn textblob emoji contractions xgboost matplotlib
"""
# pip install xgboost
# pip install nltk
# pip install emoji
# pip install contractions
# pip install textblob

import re
import string
import pandas as pd
import nltk
import emoji
import contractions
 

from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
 

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
 
import matplotlib.pyplot as plt
 

nltk.download("stopwords")
nltk.download("wordnet")
 
# Create sample dataset
# -------------------------
pos = ["I love this product 😊","Amazing service!","Excellent quality","Very happy","Worth buying"]*10
neg = ["I hate this product 😡","Bad service","Poor quality","Very disappointed","Waste of money"]*10
 
reviews = pos + neg
labels = [1]*50 + [0]*50

reviews = pos + neg

 

df = pd.DataFrame({"review":reviews,"sentiment":labels})
df.to_csv("sentiment_data.csv",index=False)
 

stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

 

def clean_text(text):
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" "," "))
    text = contractions.fix(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text) 
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", "EMAIL", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("","",string.punctuation))
    text = str(TextBlob(text).correct())
    words=[]
    for w in text.split():
        if w not in stop_words and len(w)>2:
            words.append(lemm.lemmatize(w))
    return " ".join(words)
 

df["clean_review"]=df["review"].apply(clean_text)
 
df.to_csv("cleaned_sentiment.csv",index=False)

tfidf=TfidfVectorizer(max_features=100)
X=tfidf.fit_transform(df["clean_review"])
y=df["sentiment"]

 
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y)
 
log=LogisticRegression(max_iter=500)
dt=DecisionTreeClassifier(random_state=42)
xgb=XGBClassifier(eval_metric="logloss",random_state=42)

grid=GridSearchCV(dt,{"max_depth":[2,3,5],"criterion":["gini","entropy"]},cv=3)
grid.fit(X_train,y_train)
best_dt=grid.best_estimator_

 
vote=VotingClassifier(
    estimators=[("lr",log),("dt",best_dt),("xgb",xgb)],
    voting="soft"
)

models={
    "Logistic Regression":log,
    "Decision Tree":best_dt,
    "XGBoost":xgb,
    "Voting":vote
}

for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    prob=model.predict_proba(X_test)[:,1]
    print("\\n",name)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))
    fpr,tpr,_=roc_curve(y_test,prob)
    score=auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,label=f"AUC={score:.2f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(name+" ROC Curve")
    plt.legend()
    plt.show()
    
 

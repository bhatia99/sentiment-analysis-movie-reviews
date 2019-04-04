import nltk
from nltk.corpus import movie_reviews
import numpy as np
import pandas as pd
import re

documents = [(list(movie_reviews.sents(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#flattening the list i.e. converting all sublists in a record to a songle list
main_list = []
for i in range(0,2000):
    flat_list = []
    for sublist in documents[i][0]:
        for item in sublist:
            flat_list.append(item)
    main_list.append(' '.join(flat_list))

ratings = []
for i in range(0,2000):
    ratings.append(documents[i][1])

#creating a dataframe
dataset = pd.DataFrame(main_list,columns=['Review'])
dataset['Ratings'] = ratings

#mapping 1 to pos and 0 to neg
dataset['Ratings'] = dataset['Ratings'].map(dict(pos=1, neg=0))

#shuffling the data or else data would be skewed
dataset = dataset.sample(frac=1,random_state=0).reset_index(drop=True)

#cleaning and stemming data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,2000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#finding total number of words, around 1.58 Million
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())    
word_count = len(all_words)

#creating bag of words    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1300000)
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = []
models.append(['NVB',GaussianNB()])
models.append(['RF',RandomForestClassifier(n_estimators = 100, criterion = 'entropy')])
models.append(['LR', LogisticRegression(solver='liblinear')])
models.append(['KNN', KNeighborsClassifier()])
models.append(['SVM', SVC(kernel = 'linear')])

# evaluate each model in turn
from sklearn.metrics import accuracy_score
results = []

for name, model in models:
    classifier = model
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    results.append((name,score))
    
results = pd.DataFrame(results,columns=['Classifier','Accuracy'])
print(results)

#Logistic Regression had the highest accuracy
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cf = classification_report(y_test, y_pred)
print('Confusion matrix of LR')
print(cm)
print('Classification report of LR')
print(cf)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import re

# Bu veri eşit olarak 25 bini eğitim ve 25 bini ise sınıflandırıcıyı eğitmek amacıyla bölündü.
# Ayrıca, her bir set 12.5 bin pozitif ve 12.5 bin negatif yorum içeriyor.

reviews_train_clean = []
for line in open('./movie_data/full_train.txt', 'r', encoding="utf-8"):
    reviews_train_clean.append(line.strip())

reviews_test_clean = []
for line in open('./movie_data/full_test.txt', 'r', encoding="utf-8"):
    reviews_test_clean.append(line.strip())


# Burada movie_data dosyası içindeki verilerimizi diziye atayabildik
# print(reviews_train[5])

target = [1 if i < 12500 else 0 for i in range(25000)]

# BINARY ATTRIBUTE
# Liste içindeki cümleleri ayırt ederken binary attribute' ı bir kelime birden fazla kez tekrar ediyorsa
# bu attribute ın true olması bunu engeller false olması ise tekrar eden kelimeden daha fazla varmış gibi gösterir.

# NGRAM_RANGE ATTRIBUTE
# ngram_range ise verilen aralığa göre örneğin bir yorum “didn’t love movie" gibi bir cümle içerse buradaki love
# kelimesi cümleyi pozitif gibi algılamamıza neden olur ancak cümle negatif bir anlam içermmektedir.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

stop_words = ['and', 'or']
ngram_vectorizer = CountVectorizer(max_features=1500, binary=True, stop_words=stop_words)

ngram_vectorizer.fit(reviews_train_clean)

X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size=0.8)

final = LinearSVC(C=0.01)
final.fit(X, target)

from sklearn.linear_model import LogisticRegression

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print("Accuracy for C=%s: %s"
          % (c, accuracy_score(y_val, lr.predict(X_val))))

# Accuracy for C=0.01: 0.88416
# Accuracy for C=0.05: 0.892
# Accuracy for C=0.25: 0.89424
# Accuracy for C=0.5: 0.89456
# Accuracy for C=1: 0.8944

final_ngram = LogisticRegression(C=0.5)
final_ngram.fit(X, target)
print("Final Accuracy: %s"
      % accuracy_score(target, final_ngram.predict(X_test)))


X_test = ngram_vectorizer.transform(["i don't like it"])
print("'i don't like it' cümleciği için tahmin : ",final_ngram.predict(X_test));


'''
import pickle

Model Oluşturma
with open('sentiment_analysis', 'wb') as picklefile:
    pickle.dump(ngram_vectorizer,picklefile)

Model Açma
with open('sentiment_analysis', 'rb') as training_model:
    model = pickle.load(training_model)

data = model.fit_transform(["This is so bad"])
data = pd.DataFrame(columns=list(model.get_feature_names()))

print(data)'''





import json

import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default='train.json')
    args = parser.parse_args()
    with open(args.t, 'r') as f:
        data = json.load(f)
    text = []
    result = []

    for i in range(20000):
        text.append(data[i]['data'])
        result.append(data[i]['label'])

    # for e in data:
    #     text.append(e['data'])
    #     result.append(e['label'])

    vectorizer = TfidfVectorizer()
    train_v = vectorizer.fit_transform(text)

    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(text)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    svm = LinearSVC()
    # svm.fit(X_train_tfidf, result)
    svm.fit(train_v, result)

    inp = []
    r=[]
    for i in range(5000):
        inp.append(data[i+20000]['data'])
        r.append(data[i+20000]['label'])
    # file = open('testdataexample')
    # information = file.readline()
    # triminfor = information.split('"')
    # for i in range(len(triminfor) // 2):
    #     inp.append(triminfor[2 * i + 1])

    # X_new_counts = count_vect.fit_transform(inp)
    # X_new_tfidf = tfidf_transformer.fit_transform(X_new_counts)
    test_v = vectorizer.transform(inp)

    # predicted_svm = svm.predict(X_new_tfidf)
    predicted_svm = svm.predict(test_v)

    print(numpy.mean(predicted_svm==r))

import json
import random

import numpy
from sklearn.svm import LinearSVC
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

learn = 20000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default='train.json')
    args = parser.parse_args()
    with open(args.t, 'r') as f:
        data = json.load(f)
    text = []
    result = []

    random.shuffle(data)
    for i in range(learn):
        text.append(data[i]['data'])
        result.append(data[i]['label'])

    # for e in data:
    #     text.append(e['data'])
    #     result.append(e['label'])

    vectorizer = TfidfVectorizer()
    train_v = vectorizer.fit_transform(text)

    svm = LinearSVC(C=1)
    svm.fit(train_v, result)

    inp = []
    r = []
    for i in range(25000 - learn):
        inp.append(data[i + learn]['data'])
        r.append(data[i + learn]['label'])
    # for i in range(learn):
    #     inp.append(data[i]['data'])
    #     r.append(data[i]['label'])

    # file = open('testdataexample')
    # information = file.readline()
    # triminfor = information.split('"')
    # for i in range(len(triminfor) // 2):
    #     inp.append(triminfor[2 * i + 1])

    test_v = vectorizer.transform(inp)

    predicted_svm = svm.predict(test_v)

    print(numpy.mean(predicted_svm == r))

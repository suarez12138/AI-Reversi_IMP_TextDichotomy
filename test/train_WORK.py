import json
import random

import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
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

    for e in data:
        text.append(e['review'])
        result.append(e['sentiment'])

    with open('test.json', 'r') as f:
        data = json.load(f)
    text2 = []
    result2 = []
    for e in data:
        text2.append(e['review'])
        result2.append(e['sentiment'])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    train_v = vectorizer.fit_transform(text)

    # text_clf = Pipeline([('clf', LinearSVC(multi_class='crammer_singer', class_weight='balanced'))])
    # parameters = {
    #     # 'Tfi__use_idf': (True, False),
    #     # 'Tfi__smooth_idf': (True, False),
    #     # 'Tfi__sublinear_tf': (True, False),
    #     # 'Tfi__max_df': (1),
    #     # 'Tfi__min_df': (1, 0.7),
    #     # 'Tfi__ngram_range': [(1, 2)],
    #     # 'clf__penalty': ('l1', 'l2'),
    #     # 'clf__loss': ('hinge', 'squared_hinge'),
    #     # 'clf__multi_class': ('ovr', 'crammer_singer'),
    #     # 'clf__dual': (True, False),
    #     # 'clf__class_weight': ('balanced', None),
    #     'clf__C': (1.06, 1.055, 1.05, 1.04, 1.045),
    #     # 'clf__fit_intercept': (True, False),
    #     # 'clf__tol': (1e-4, 1.03e-4, 1.1e-4, 1.05e-4, 9.9e-5, 9.95e-5),
    #     # 'clf__dual': (True, False)
    #     'clf__intercept_scaling': (1.4, 1.395, 1.405)
    # }
    # gs_clf = GridSearchCV(text_clf, parameters)
    # gs_clf = gs_clf.fit(train_v, result)
    # print(gs_clf.best_score_)
    # print(gs_clf.best_params_)

    svm = LinearSVC(C=1.053, intercept_scaling=1.41,multi_class='crammer_singer',class_weight='balanced')
    svm.fit(train_v, result)

    text_2 = vectorizer.transform(text2)
    predicted_svm = svm.predict(text_2)
    print(numpy.mean(predicted_svm == result2))

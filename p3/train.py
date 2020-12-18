import json
import argparse
import zipfile

import joblib
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default='train.json')
    args = parser.parse_args()
    with open(args.t, 'r') as f:
        data = json.load(f)
    text = []
    result = []

    for e in data:
        text.append(e['data'])
        result.append(e['label'])
    vectorizer = TfidfVectorizer()
    train_v = vectorizer.fit_transform(text)
    svm = LinearSVC()
    svm.fit(train_v, result)

    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(svm, 'svm.pkl')
    f = zipfile.ZipFile('model', 'w', zipfile.ZIP_DEFLATED)
    f.write('vectorizer.pkl')
    f.write('svm.pkl')
    f.close()
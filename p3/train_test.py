import argparse
import json
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default='train.json')
    parser.add_argument('-i', type=str, default='testdataexample')
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

    inp = []
    file = open(args.i)
    information = file.readline()
    triminfor = information.split('"')
    for i in range(len(triminfor) // 2):
        inp.append(triminfor[2 * i + 1])
    test_v = vectorizer.transform(inp)
    predicted_svm = svm.predict(test_v)
    with open("output.txt", "w") as f:
        for i in range(len(predicted_svm)):
            if i == len(predicted_svm) - 1:
                f.write(str(predicted_svm[i]))
            else:
                f.write(str(predicted_svm[i]) + '\n')
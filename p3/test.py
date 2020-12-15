import argparse
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='testdataexample')
    parser.add_argument('-m', type=str, default='none')
    args = parser.parse_args()
    if args.m == 'none':
        model2 = joblib.load('model2')
        model = joblib.load('model')
    else:
        model = joblib.load(args.m)
    inp = []
    file = open(args.i)
    information = file.readline()
    triminfor = information.split('"')
    for i in range(len(triminfor) // 2):
        inp.append(triminfor[2 * i + 1])

    test_v = model2.transform(inp)

    predicted_svm = model.predict(test_v)

    with open("output.txt", "w") as f:
        for i in range(len(predicted_svm)):
            if i == len(predicted_svm) - 1:
                f.write(str(predicted_svm[i]))
            else:
                f.write(str(predicted_svm[i]) + '\n')

import pandas as pd
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import datasets


# Trasform the sklearn's classification_report into a csv file in the path
def classification_report_csv(report: object, path: object) -> object:
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:
        row = {}
        print(line)
        row_data = line.split()
        print(row_data)
        row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(path, index=False)


if __name__ == '__main__':
    print('[INFO] Testing...')
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)
    predictions=clf.predict(X)
    report = classification_report(y, predictions)
    print(report)
    classification_report_csv(report, 'test/csv_test.csv')

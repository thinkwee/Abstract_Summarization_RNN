import gensim
import os
from sklearn import svm
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def file_preprocessing(file_name):
    file_output = open('./data/' + file_name + '.txt', 'w')
    with open('./data/sample.' + file_name + '.txt', 'r') as f:
        for line in f:
            if not line.__contains__('review') and line != "\n" and len(line) > 10:
                file_output.write(line)
    f.close()
    file_output.close()
    print(file_name + " preprocessing complete")


class SVM(object):
    def __init__(self, trainset, testset):
        self.trainset = trainset
        self.testset = testset
        self.file_train = open(self.trainset, 'r+')
        self.file_test = open(self.testset, 'r+')
        self.train_data = np.loadtxt(self.file_train)
        self.train_x = self.train_data[:, 1:]
        self.train_y = self.train_data[:, 0]
        self.test_data = np.loadtxt(self.file_test)
        self.test_x = self.test_data[:, :]
        self.clf = svm.SVC()

    def Normalization(self):
        self.train_x = preprocessing.minmax_scale(self.train_x, feature_range=(-1, 1))
        self.test_x = preprocessing.minmax_scale(self.test_x, feature_range=(-1, 1))

    def Fitclf(self):
        self.clf.fit(self.train_x, self.train_y)

    def Predict(self):
        self.result = self.clf.predict(self.test_x)
        return self.result

    def SaveModel(self):
        joblib.dump(self.clf, './SVM/train_model')

    def LoadModel(self):
        self.clf = joblib.load('./SVM/train_model')


def train_SVM():
    train_data = "./data/doc_vector.txt"
    test_data = "./data/doc_vector_test.txt"

    classifier = SVM(train_data, test_data)
    # print(classifier.train_x)
    # print(classifier.train_y)
    classifier.Normalization()
    # classifier.Fitclf()
    # classifier.SaveModel()
    classifier.LoadModel()
    print("SVM training complete")
    result = classifier.Predict()
    standard = []
    for i in range(100):
        standard.append(1)
        standard.append(-1)
    target_name = ['negative', 'positive']
    print(standard)
    print(result)
    print(classification_report(standard, result, target_names=target_name))


def train_Doc2Vec():
    input_file_p = open("./data/positive.txt", "r")
    input_file_n = open("./data/negative.txt", "r")
    output_together = open("./data/together.txt", "w")
    p = []
    n = []
    for line in input_file_p:
        p.append(line)
    for line in input_file_n:
        n.append(line)
    for i in range(10200):
        output_together.write(p[i])
        output_together.write(n[i])
    output_together.close()
    input_file_n.close()
    input_file_p.close()

    input_file = open("./data/together.txt", "r")
    output_file = open("./data/doc_vector.txt", "w")
    sentence = gensim.models.doc2vec.TaggedLineDocument(input_file)
    model = gensim.models.Doc2Vec(sentence, vector_size=100, window=5)
    print("Doc2Vec training completed")
    checkpoint = "./doc2vec/vec_model"
    model.save(checkpoint)

    for i in range(20000):
        if i % 2 == 0:
            output_file.write('1 ')
        else:
            output_file.write('-1 ')
        for j in range(100):
            output_file.write(str(model.docvecs[i][j]) + ' ')
        output_file.write('\n')

    print('vector output completed')
    input_file.close()
    output_file.close()
    test_file = open("./data/doc_vector_test.txt", "w")
    for i in range(20000, 20200):
        for j in range(100):
            test_file.write(str(model.docvecs[i][j]) + ' ')
        test_file.write('\n')
    test_file.close()
    print('vector test output completed')


# file_preprocessing("positive")
# file_preprocessing("negative")

# train_Doc2Vec()

train_SVM()

import sentiwordnet
import nltk
from sklearn import svm
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import gensim


def make_np_vector():
    file_p = open("./data/positive.txt", "r")
    file_n = open("./data/negative.txt", "r")
    file_vec = open("./data/np_dict_vector.txt", "w")
    file_vec_test = open("./data/np_dict_vector_test.txt", "w")
    sentence_p = []
    sentence_n = []

    for line in file_p:
        sentence_p.append(line)
    file_p.close()
    for line in file_n:
        sentence_n.append(line)
    file_n.close()

    net_path = "./data/SentiWordNet.txt"
    np_dict = sentiwordnet.SentiWordNet(net_path)
    np_dict.infoextract()
    for i in range(10100):
        text_p = nltk.word_tokenize(sentence_p[i])
        pos_info_p = nltk.pos_tag(text_p)
        text_n = nltk.word_tokenize(sentence_n[i])
        pos_info_n = nltk.pos_tag(text_n)
        vector_p = sentiwordnet.make_np_vector(np_dict, pos_info_p)
        vector_n = sentiwordnet.make_np_vector(np_dict, pos_info_n)
        if i < 10000:
            file_vec.write("1 ")
            for j in range(6):
                file_vec.write(str(vector_p[j]) + " ")
            file_vec.write("\n")
            file_vec.write("-1 ")
            for j in range(6):
                file_vec.write(str(vector_n[j]) + " ")
            file_vec.write("\n")
        else:
            for j in range(6):
                file_vec_test.write(str(vector_p[j]) + " ")
            file_vec_test.write("\n")
            for j in range(6):
                file_vec_test.write(str(vector_n[j]) + " ")
            file_vec_test.write("\n")

    file_vec_test.close()
    print("vector output complete")


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
        self.clf = svm.SVC(probability=True)

    def Normalization(self):
        self.train_x = preprocessing.minmax_scale(self.train_x, feature_range=(-1, 1))
        self.test_x = preprocessing.minmax_scale(self.test_x, feature_range=(-1, 1))

    def Fitclf(self):
        self.clf.fit(self.train_x, self.train_y)

    def Predict(self):
        self.result = self.clf.predict(self.test_x)
        # self.result_prob = self.clf.predict_proba(self.test_x)
        self.result_prob = self.clf.decision_function(self.test_x)
        return self.result, self.result_prob

    def SaveModel(self):
        joblib.dump(self.clf, './SVM/train_model_dict')

    def LoadModel(self):
        self.clf = joblib.load('./SVM/train_model_dict')


def train_SVM():
    train_data = "./data/np_dict_vector.txt"
    test_data = "./data/np_dict_vector_test.txt"

    classifier = SVM(train_data, test_data)
    classifier.Normalization()
    # classifier.Fitclf()
    # classifier.SaveModel()
    classifier.LoadModel()
    print("SVM training complete")
    result, result_prob = classifier.Predict()

    standard = []
    for i in range(100):
        standard.append(1)
        standard.append(-1)
    target_name = ['negative', 'positive']

    # result_depend_on_prob = []
    # for i in range(200):
    #     if result_prob[i][0] > 0.5:
    #         result_depend_on_prob.append(-1)
    #     else:
    #         result_depend_on_prob.append(1)

    print(classification_report(standard, result, target_names=target_name))
    # print(classification_report(standard, result_depend_on_prob, target_names=target_name))

    for i in range(200):
        if result[i] == standard[i]:
            flag = True
        else:
            flag = False
        if not flag and result[i] == -1:
            print(i, result_prob[i], standard[i], flag)


# make_np_vector()
train_SVM()

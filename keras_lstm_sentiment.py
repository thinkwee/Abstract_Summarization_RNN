from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk  # 用来分词
import collections  # 用来统计词频
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

MAX_FEATURES = 5000
MAX_SENTENCE_LENGTH = 60

maxlen = 0  # 句子最大长度
word_freqs = collections.Counter()  # 词频
num_recs = 0  # 样本数
totallen = 0
with open('./data/keras_pn_train.txt', 'r+') as f:
    for line in f:
        label, sentence = line.strip().split("-!-!-!")
        words = nltk.word_tokenize(sentence.lower())
        totallen += len(words)
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print("avg_len ", totallen / 20000)
print('max_len ', maxlen)
print('nb_words ', len(word_freqs))

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v: k for k, v in word2index.items()}


def train():
    # nltk.download('punkt')
    # input_file_p = open("./data/positive.txt", "r")
    # input_file_n = open("./data/negative.txt", "r")
    # output_together = open("./data/keras_pn_train.txt", "w+")
    # p = []
    # n = []
    # for line in input_file_p:
    #     p.append(line)
    # for line in input_file_n:
    #     n.append(line)
    # for i in range(10000):
    #     output_together.write('1\t' + p[i])
    #     output_together.write('0\t' + n[i])
    # output_together.close()
    # input_file_n.close()
    # input_file_p.close()
    # print("p/n copora added")
    #
    X = np.empty(num_recs, dtype=list)
    y = np.zeros(num_recs)
    i = 0
    with open('./data/keras_pn_train.txt', 'r+') as f:
        for line in f:
            label, sentence = line.strip().split("-!-!-!")
            words = nltk.word_tokenize(sentence.lower())
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs
            y[i] = int(label)
            i += 1
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=666)

    EMBEDDING_SIZE = 128
    HIDDEN_LAYER_SIZE = 128

    # model = Sequential()
    # model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
    # model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.5, recurrent_dropout=0.5))
    # # model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2)))
    # model.add(Dense(1))
    # model.add(Activation("sigmoid"))
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    #
    # file_path = "./keras_model/weights-improvement-{epoch:02d}-{val_acc:.5f}.hdf5"
    # checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    #
    # BATCH_SIZE = 32
    # NUM_EPOCHS = 10
    # model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest),
    #           callbacks=callbacks_list)
    #
    # score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    # print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))

    model = load_model("./keras_model/weights-improvement-04-0.74100.hdf5")
    print(Xtest)
    print('{}   {}      {}'.format('预测', '真实', '句子'))
    for i in range(20):
        idx = np.random.randint(len(Xtest))
        xtest = Xtest[idx].reshape(1, MAX_SENTENCE_LENGTH)
        print(xtest)
        ylabel = ytest[idx]
        ypred = model.predict(xtest)[0][0]
        sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
        print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))

    # INPUT_SENTENCES = []
    # with open("./data/keras_pn_train.txt") as f:
    #     for line in f:
    #         INPUT_SENTENCES.append(line)
    # XX = np.empty(len(INPUT_SENTENCES), dtype=list)
    # i = 0
    # for sentence in INPUT_SENTENCES:
    #     words = nltk.word_tokenize(sentence.lower())
    #     seq = []
    #     for word in words:
    #         if word in word2index:
    #             seq.append(word2index[word])
    #         else:
    #             seq.append(word2index['UNK'])
    #     XX[i] = seq
    #     i += 1
    #
    # XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
    # labels = [int(round(x[0])) for x in model.predict(XX)]
    # label2word = {1: '积极', 0: '消极'}
    # for i in range(len(INPUT_SENTENCES)):
    #     print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))


def predict(file_name):
    num_recs = 100000
    X = np.empty(num_recs, dtype=list)
    i = 0
    with open('./data/' + file_name + '_middle.txt', 'r+') as f:
        for line in f:
            words = nltk.word_tokenize(line.strip().lower())
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs
            i += 1
            if i == 100000:
                break
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
    print("word2index completed")

    model = load_model("./keras_model/weights-improvement-04-0.74100.hdf5")
    output_file = open("./data/keras_lstm_sen_" + file_name + ".txt", "w")
    for i in range(100000):
        if i % 1000 == 0:
            print("%d/100  completed" % (i / 1000))
        x = X[i].reshape(1, MAX_SENTENCE_LENGTH)
        ypred = int(round(model.predict(x)[0][0]))
        output_file.writelines(str(ypred) + "\n")
    print("output completed")


def remake_middle_copora():
    file_article = open("./data/keras_lstm_sen_article.txt", "r")
    file_headline = open("./data/keras_lstm_sen_headline.txt", "r")
    article_np = []
    headline_np = []
    for line in file_article:
        article_np.append(int(line))
    for line in file_headline:
        headline_np.append(int(line))
    file_article.close()
    file_headline.close()

    file_article = open("./data/article_middle.txt", "r")
    file_headline = open("./data/headline_middle.txt", "r")

    article = []
    headline = []

    for line in file_article:
        article.append(line)
    for line in file_headline:
        headline.append(line)

    file_headline.close()
    file_article.close()

    file_article_output = open("./data/article_middle_sen.txt", "w")
    file_headline_output = open("./data/headline_middle_sen.txt", "w")
    file_sen_output = open("./data/middle_sen.txt", "w")
    count_p = 0
    count_n = 0
    for i in range(100000):
        if article_np[i] == headline_np[i]:
            file_article_output.writelines(article[i])
            file_headline_output.writelines(headline[i])
            if article_np[i] == 1:
                count_p += 1
            elif article_np[i] == 0:
                count_n += 1
            file_sen_output.write(str(article_np[i]) + " ")

    print(count_p, count_n, count_n + count_p)

# predict("article")
# train()
# remake_middle_copora()

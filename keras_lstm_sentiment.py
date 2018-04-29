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
maxlen = 0  # 句子最大长度
word_freqs = collections.Counter()  # 词频
num_recs = 0  # 样本数
totallen = 0
with open('./data/keras_np_train.txt', 'r+') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
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

MAX_FEATURES = 3000
MAX_SENTENCE_LENGTH = 40

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v: k for k, v in word2index.items()}

X = np.empty(num_recs, dtype=list)
y = np.zeros(num_recs)
i = 0
with open('./data/keras_np_train.txt', 'r+') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
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

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2333)

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 128

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
# model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

file_path = "./keras_model/weights-improvement-{epoch:02d}-{val_acc:.5f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

BATCH_SIZE = 32
NUM_EPOCHS = 10
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest),
          callbacks=callbacks_list)

score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('预测', '真实', '句子'))
for i in range(20):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1, MAX_SENTENCE_LENGTH)
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

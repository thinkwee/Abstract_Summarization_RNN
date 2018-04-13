from word2vec.w2v import w2v
import logging.config
import pickle
from seq2seq import Seq2seqModel
from pre_process import *
from numpy import *

"""Log Configuration"""
LOG_FILE = './log/train.log'
handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('trainlogger')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)

"""Hyper Parameters(Seq2Seq train)"""
VOCAB_SIZE = 3650
EMBED_SIZE = 128
ENCODER_HIDEEN_UNITS = 512
DECODER_HIDDEN_UNITS = 1024
BATCH_SIZE = 32
ENCODER_LAYERS = 1
EPOCH = 1000
NUM_TRAIN_STEPS = 215
SKIP_STEPS = 50
LEARNING_RATE_INITIAL = 0.1
KEEP_PROB = 0.2
CONTINUE_TRAIN = 0

"""Hyper Parameters(Seq2seq infer)"""
BATCH_SIZE_INFER = 100
EPOCH_INFER = 1
NUM_TRAIN_STEPS_INFER = 1

"""Hyper Parameters(Word2Vec)"""
NUM_SAMPLED = 64
LEARNING_RATE_W2V = 1.0
SKIP_WINDOWS = 1
DATA_NAME_W2V = 'traintext.zip'
NUM_TRAIN_STEPS_W2V = 17000
SKIP_STEPS_W2V = 100


def build_embed_matrix():
    w2vmodel = w2v(vocab_size=VOCAB_SIZE,
                   embed_size=EMBED_SIZE,
                   batch_size=BATCH_SIZE,
                   num_sampled=NUM_SAMPLED,
                   learning_rate=LEARNING_RATE_W2V,
                   skip_windows=SKIP_WINDOWS,
                   data_name=DATA_NAME_W2V,
                   num_train_steps=NUM_TRAIN_STEPS_W2V,
                   skip_steps=SKIP_STEPS_W2V)
    w2vmodel.build_graph()
    return w2vmodel.train()


def save_embed_matrix(embed_matrix, one_hot_dictionary, one_hot_dictionary_index):
    output = open('./save/embed_matrix.pkl', 'wb')
    pickle.dump(embed_matrix, output)
    output.close()
    output = open('./save/one_hot_dictionary.pkl', 'wb')
    pickle.dump(one_hot_dictionary, output)
    output.close()
    output = open('./save/one_hot_dictionary_index.pkl', 'wb')
    pickle.dump(one_hot_dictionary_index, output)
    output.close()


def load_embed_matrix():
    pkl_file = open('./save/embed_matrix.pkl', 'rb')
    embed_matrix = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('./save/one_hot_dictionary.pkl', 'rb')
    one_hot_dictionary = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('./save/one_hot_dictionary_index.pkl', 'rb')
    one_hot_dictionary_index = pickle.load(pkl_file)
    pkl_file.close()
    return embed_matrix, one_hot_dictionary, one_hot_dictionary_index


def train(embed_matrix, one_hot_dictionary, continue_train, start_token_id, end_token_id):
    print("train mode")

    single_generate = one_hot_generate(one_hot_dictionary=one_hot_dictionary,
                                       epoch=EPOCH,
                                       is_train=1)
    batches = get_batch(batch_size=BATCH_SIZE,
                        iterator=single_generate)
    logger.debug("batch generated")

    seq2seq_train = Seq2seqModel(vocab_size=VOCAB_SIZE,
                                 embed_size=EMBED_SIZE,
                                 encoder_hidden_units=ENCODER_HIDEEN_UNITS,
                                 decoder_hidden_units=DECODER_HIDDEN_UNITS,
                                 batch_size=BATCH_SIZE,
                                 embed_matrix_init=embed_matrix,
                                 encoder_layers=ENCODER_LAYERS,
                                 learning_rate_initial=LEARNING_RATE_INITIAL,
                                 keep_prob=KEEP_PROB,
                                 core="bgru",
                                 start_token_id=start_token_id,
                                 end_token_id=end_token_id
                                 )
    seq2seq_train.build_graph()
    print("the model has been built")

    if continue_train == 0:
        print("first training")
        seq2seq_train.first_train(epoch_total=EPOCH,
                                  num_train_steps=NUM_TRAIN_STEPS,
                                  batches=batches,
                                  skip_steps=SKIP_STEPS)
    else:
        print("continue training")
        seq2seq_train.continue_train(epoch_total=EPOCH,
                                     num_train_steps=NUM_TRAIN_STEPS,
                                     batches=batches,
                                     skip_steps=SKIP_STEPS)


def test(embed_matrix, one_hot_dictionary, one_hot_dictionary_index, start_token_id, end_token_id):
    print("infer mode")
    single_generate = one_hot_generate(one_hot_dictionary,
                                       epoch=EPOCH_INFER,
                                       is_train=0)
    batches = get_batch(batch_size=BATCH_SIZE_INFER,
                        iterator=single_generate)
    logger.debug("batch generated")

    seq2seq_infer = Seq2seqModel(vocab_size=VOCAB_SIZE,
                                 embed_size=EMBED_SIZE,
                                 encoder_hidden_units=ENCODER_HIDEEN_UNITS,
                                 decoder_hidden_units=DECODER_HIDDEN_UNITS,
                                 encoder_layers=ENCODER_LAYERS,
                                 batch_size=BATCH_SIZE_INFER,
                                 learning_rate_initial=LEARNING_RATE_INITIAL,
                                 embed_matrix_init=embed_matrix,
                                 keep_prob=KEEP_PROB,
                                 core="bgru",
                                 start_token_id=start_token_id,
                                 end_token_id=end_token_id)
    seq2seq_infer.build_graph()
    seq2seq_infer.test(num_train_steps=NUM_TRAIN_STEPS_INFER,
                       batches=batches,
                       one_hot=one_hot_dictionary_index)
    logger.debug("seq2seq model tested")


def main():
    # print("test word2vec model")
    # embed_matrix, one_hot_dictionary, one_hot_dictionary_index = build_embed_matrix()
    # logger.debug("w2v finished")
    #
    # save_embed_matrix(embed_matrix, one_hot_dictionary, one_hot_dictionary_index)
    # logger.debug("w2v saved")

    embed_matrix, one_hot_dictionary, one_hot_dictionary_index = load_embed_matrix()
    logger.debug("w2v restored")

    # print(one_hot_dictionary)
    # print(one_hot_dictionary_index)
    start_token_id = one_hot_dictionary['_GO']
    end_token_id = one_hot_dictionary['_EOS']
    # train(embed_matrix=embed_matrix, one_hot_dictionary=one_hot_dictionary, continue_train=CONTINUE_TRAIN,
    #       start_token_id=start_token_id, end_token_id=end_token_id)
    test(embed_matrix=embed_matrix, one_hot_dictionary=one_hot_dictionary,
         one_hot_dictionary_index=one_hot_dictionary_index, start_token_id=start_token_id, end_token_id=end_token_id)


if __name__ == '__main__':
    # t = timeit('main()', 'from __main__ import main', number=1)
    # print(t)
    main()

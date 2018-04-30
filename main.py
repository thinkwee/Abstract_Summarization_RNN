from word2vec.w2v import w2v
import logging.config
from seq2seq import Seq2seqModel
import pre_process as pre
import pre_process_senti as pre_senti
from numpy import *
import sys
import pickle

"""Log Configuration"""
LOG_FILE = './log/train.log'
handler = logging.FileHandler(LOG_FILE, mode='w')
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('trainlogger')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

"""Hyper Parameters(Seq2Seq train)"""
VOCAB_SIZE = 3000
EMBED_SIZE = 256
ENCODER_HIDEEN_UNITS = 512
DECODER_HIDDEN_UNITS = 1024
LEARNING_RATE_INITIAL = 0.1
BATCH_SIZE = 32
RNN_LAYERS = 2
EPOCH = 1000
NUM_TRAIN_STEPS = 1850
SKIP_STEPS = 200
KEEP_PROB = 0.5
CONTINUE_TRAIN = 0
GRAD_CLIP = 1.0

"""Hyper Parameters(Seq2seq infer)"""
BATCH_SIZE_INFER = 32
EPOCH_INFER = 1
NUM_TRAIN_STEPS_INFER = 1

"""Hyper Parameters(Word2Vec)"""
NUM_SAMPLED = 32
LEARNING_RATE_W2V = 0.5
SKIP_WINDOWS = 1
DATA_NAME_W2V = 'traintext.zip'
NUM_TRAIN_STEPS_W2V = 10000
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


def train(embed_matrix, one_hot_dictionary, start_token_id, end_token_id):
    print("train mode")

    single_generate = pre_senti.one_hot_generate(one_hot_dictionary=one_hot_dictionary,
                                                 epoch=EPOCH,
                                                 is_train=1)
    batches = pre_senti.get_batch(batch_size=BATCH_SIZE,
                                  iterator=single_generate)
    logger.debug("batch generated")

    seq2seq_train = Seq2seqModel(vocab_size=VOCAB_SIZE,
                                 embed_size=EMBED_SIZE,
                                 encoder_hidden_units=ENCODER_HIDEEN_UNITS,
                                 decoder_hidden_units=DECODER_HIDDEN_UNITS,
                                 batch_size=BATCH_SIZE,
                                 embed_matrix_init=embed_matrix,
                                 learning_rate_initial=LEARNING_RATE_INITIAL,
                                 keep_prob=KEEP_PROB,
                                 rnn_core="bgru_attetion",
                                 start_token_id=start_token_id,
                                 end_token_id=end_token_id,
                                 num_layers=RNN_LAYERS,
                                 grad_clip=GRAD_CLIP,
                                 is_continue=CONTINUE_TRAIN)
    seq2seq_train.build_graph()
    print("the model has been built")

    seq2seq_train.train(epoch_total=EPOCH,
                        num_train_steps=NUM_TRAIN_STEPS,
                        batches=batches,
                        skip_steps=SKIP_STEPS)


def test(embed_matrix, one_hot_dictionary, one_hot_dictionary_index, start_token_id, end_token_id):
    print("infer mode")
    single_generate = pre_senti.one_hot_generate(one_hot_dictionary,
                                                 epoch=EPOCH_INFER,
                                                 is_train=0)
    batches = pre_senti.get_batch(batch_size=BATCH_SIZE_INFER,
                                  iterator=single_generate)
    logger.debug("batch generated")

    seq2seq_infer = Seq2seqModel(vocab_size=VOCAB_SIZE,
                                 embed_size=EMBED_SIZE,
                                 encoder_hidden_units=ENCODER_HIDEEN_UNITS,
                                 decoder_hidden_units=DECODER_HIDDEN_UNITS,
                                 batch_size=BATCH_SIZE_INFER,
                                 learning_rate_initial=LEARNING_RATE_INITIAL,
                                 embed_matrix_init=embed_matrix,
                                 keep_prob=KEEP_PROB,
                                 rnn_core="bgru_attetion",
                                 start_token_id=start_token_id,
                                 end_token_id=end_token_id,
                                 num_layers=RNN_LAYERS,
                                 grad_clip=GRAD_CLIP,
                                 is_continue=0)
    seq2seq_infer.build_graph()
    seq2seq_infer.test(epoch=EPOCH_INFER,
                       num_train_steps=NUM_TRAIN_STEPS_INFER,
                       batches=batches,
                       one_hot=one_hot_dictionary_index)
    logger.debug("seq2seq model tested")


def batch_test(test_batch_num, one_hot_dictionary):
    single_generate = pre_senti.one_hot_generate(one_hot_dictionary=one_hot_dictionary,
                                                 epoch=1,
                                                 is_train=1)
    batches = pre_senti.get_batch(batch_size=32,
                                  iterator=single_generate)
    for i in range(test_batch_num):
        encoder_batch, decoder_batch, target_batch, bucket_encoder_length, bucket_decoder_length, decode_max_iter, senti_batch = next(
            batches)
        print(i)
        print(encoder_batch)
        print(decoder_batch)
        print(target_batch)
        print(bucket_encoder_length)
        print(bucket_decoder_length)
        print(decode_max_iter)
        print(senti_batch)


def main():
    option = sys.argv[1]
    if option == "-w2v":
        print("train word2vec model")
        embed_matrix, one_hot_dictionary, one_hot_dictionary_index = build_embed_matrix()
        logger.debug("w2v finished")

        save_embed_matrix(embed_matrix, one_hot_dictionary, one_hot_dictionary_index)
        logger.debug("w2v saved")
    elif option == "-train":
        embed_matrix, one_hot_dictionary, one_hot_dictionary_index = load_embed_matrix()
        logger.debug("w2v restored")
        start_token_id = one_hot_dictionary['_GO']
        end_token_id = one_hot_dictionary['_EOS']
        train(embed_matrix=embed_matrix, one_hot_dictionary=one_hot_dictionary,
              start_token_id=start_token_id, end_token_id=end_token_id)
    elif option == "-test":
        embed_matrix, one_hot_dictionary, one_hot_dictionary_index = load_embed_matrix()
        logger.debug("w2v restored")
        start_token_id = one_hot_dictionary['_GO']
        end_token_id = one_hot_dictionary['_EOS']
        test(embed_matrix=embed_matrix, one_hot_dictionary=one_hot_dictionary,
             one_hot_dictionary_index=one_hot_dictionary_index, start_token_id=start_token_id,
             end_token_id=end_token_id)
    elif option == "-check":
        embed_matrix, one_hot_dictionary, one_hot_dictionary_index = load_embed_matrix()
        print("one_hot_dictionary:")
        print(one_hot_dictionary)
        start_token_id = one_hot_dictionary['_GO']
        end_token_id = one_hot_dictionary['_EOS']
        pad_token_id = one_hot_dictionary['_PAD']
        unk_token_id = one_hot_dictionary['_UNK']
        print("_GO _EOS _PAD _UNK")
        print(start_token_id, end_token_id, pad_token_id, unk_token_id)
    elif option == "-batch":
        print("batch test")
        embed_matrix, one_hot_dictionary, one_hot_dictionary_index = load_embed_matrix()
        batch_test(1, one_hot_dictionary)
    else:
        print("wrong option")
        print("use -w2v to train word2vec embed matrix")
        print("use -train to train the seq2seq model")
        print("use -test to test the seq2seq model")
        print("use -check to check word2vec embed matrix")
        print("use -batch to test batch generalization")


if __name__ == '__main__':
    main()

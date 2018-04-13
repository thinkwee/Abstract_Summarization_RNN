import string
import numpy as np
import collections


def get_words(file_name, n):
    # get most n frequent word in file_name.txt

    filename = './data/' + file_name + '.txt'
    with open(filename) as f:
        words_box = []
        for line in f:
            words_box.extend(line.strip().split())
    word_num = collections.Counter(words_box).most_common(n)
    # print(word_num)
    mostnwords = set()
    for word, num in word_num:
        mostnwords.add(word)
    return mostnwords


def changetext(batch_size, file_name, mostnwords):
    # add <_GO>,<_EOS>,<_PAD>,<_UNK> to the file_name.txt
    # since we use tensorflow.dynaic_rnn,we don't need to pad sentences in each batch to a same length

    filename_sorted = './data/' + file_name + '_sorted.txt'
    filename_processed = './data/' + file_name + '_processed.txt'

    file_processed = open(filename_processed, 'w')

    # calculate the max sentence length in each batch
    batch_length = []
    max_length = 0
    index = 0
    for sentence in open(filename_sorted):
        index += 1
        sentence_length = len(sentence.split())
        if max_length < sentence_length:
            max_length = sentence_length
        if index == batch_size:
            batch_length.append(max_length)
            max_length = 0
            index = 0
    if index != 0:
        batch_length.append(max_length)
    # print(batch_length)

    # add <_GO>,<_EOS>,<_UNK>to the sentence
    batch_idx = 0
    index = 0
    if file_name == 'headline':
        for line in open(filename_sorted):
            newline = ""
            newline += '_GO '
            for word in line.split():
                if word not in mostnwords:
                    newline += '_UNK '
                else:
                    newline += word + ' '
            # for _ in range(batch_length[batch_idx] - len(line.split())):
            #     newline += '_PAD '
            # print("%d %d" % (batch_length[batch_idx], len(line.split())))
            index += 1
            if index == batch_size:
                batch_idx += 1
                index = 0
            newline += '_EOS\n'
            # if batch_length[batch_idx] != len(line.split()):
            file_processed.writelines(newline)
        print('pre processing headline finished')
    elif file_name == 'article':
        for line in open(filename_sorted):
            newline = ""
            for word in line.split():
                if word not in mostnwords:
                    newline += '_UNK '
                else:
                    newline += word + ' '
            # for _ in range(batch_length[batch_idx] - len(line.split())):
            #     newline += '_PAD '
            # print("%d %d" % (batch_length[batch_idx], len(line.split())))
            index += 1
            if index == batch_size:
                batch_idx += 1
                index = 0
            newline += '\n'
            # if batch_length[batch_idx] != len(line.split()):
            file_processed.writelines(newline)
        print('pre processing article finished')
    else:
        print('wrong during processing,please verify your file name')
    file_processed.close()


def get_batch(batch_size, iterator):
    while True:
        encoder_batch = []
        decoder_batch = []
        target_batch = []
        bucket_encoder_length = []
        bucket_decoder_length = []

        bucket_encoder_length_max = 0
        bucket_decoder_length_max = 0

        for index in range(batch_size):
            encoder_input_single, decoder_input_single, target_single, encoder_length_single, decoder_length_single = next(
                iterator)

            if bucket_decoder_length_max < decoder_length_single:
                bucket_decoder_length_max = decoder_length_single
            if bucket_encoder_length_max < encoder_length_single:
                bucket_encoder_length_max = encoder_length_single

            encoder_batch.append(encoder_input_single)
            decoder_batch.append(decoder_input_single)
            target_batch.append(target_single)
            # bucket_encoder_length.append(encoder_length_single)
            # bucket_decoder_length.append(decoder_length_single)
            # bucket_encoder_length_max = encoder_length_single
            # bucket_decoder_length_max = decoder_length_single

        for index in range(batch_size):
        #     len_temp = len(encoder_batch[index])
            encoder_batch[index] = np.resize(encoder_batch[index], [bucket_encoder_length_max])
        #     encoder_batch[index][len_temp + 1:] = 0
        #
        #     len_temp = len(decoder_batch[index])
            decoder_batch[index] = np.resize(decoder_batch[index], [bucket_decoder_length_max])
        #     target_batch[index][len_temp + 1:] = 0
        #
        #     len_temp = len(target_batch[index])
            target_batch[index] = np.resize(target_batch[index], [bucket_decoder_length_max])
        #     target_batch[index][len_temp + 1:] = 0

        # for index in range(batch_size):
        #     len_temp = len(encoder_batch[index])
        #     encoder_batch[index] = np.resize(encoder_batch[index], [bucket_encoder_length])
        #     encoder_batch[index][len_temp + 1:] = 0
        #
        #     len_temp = len(decoder_batch[index])
        #     decoder_batch[index] = np.resize(decoder_batch[index], [bucket_decoder_length])
        #     target_batch[index] = np.resize(target_batch[index], [bucket_decoder_length])
        #
        #     decoder_batch[index][len_temp + 1:] = 0
        #     target_batch[index][len_temp:] = 0
        #
        bucket_encoder_length = [bucket_encoder_length_max for _ in range(batch_size)]
        bucket_decoder_length = [bucket_decoder_length_max for _ in range(batch_size)]
        yield encoder_batch, decoder_batch, target_batch, bucket_encoder_length, bucket_decoder_length


def one_hot_generate(one_hot_dictionary, epoch, is_train):
    for i in range(epoch):
        if is_train:
            file_headline = open('./data/headline_train.txt', 'rb')
            file_article = open('./data/article_train.txt', 'rb')
        else:
            file_headline = open('./data/headline_test.txt', 'rb')
            file_article = open('./data/article_test.txt', 'rb')

        sentence_article = bytes.decode(file_article.readline())
        sentence_headline = bytes.decode(file_headline.readline())

        while sentence_article and sentence_headline:
            words_article = []
            words_headline = []
            count_article = 0
            count_headline = 0
            strip = string.whitespace + string.punctuation + "\"'"
            for word in sentence_article.split():
                word = word.strip(strip)
                words_article.append(word)
                count_article += 1
            for word in sentence_headline.split():
                word = word.strip(strip)
                words_headline.append(word)
                count_headline += 1
            one_hot_article = np.zeros([count_article], dtype=int)
            one_hot_headline_raw = np.zeros([count_headline], dtype=int)
            # one_hot_headline_input = np.zeros([count_headline - 1], dtype=int)
            # one_hot_headline_target = np.zeros([count_headline - 1], dtype=int)

            for index, word in enumerate(words_article):
                one_hot_article[index] = one_hot_dictionary[word] if word in one_hot_dictionary else 0

            for index, word in enumerate(words_headline):
                one_hot_headline_raw[index] = one_hot_dictionary[word] if word in one_hot_dictionary else 0

            # raw: <_GO> V V V V V V <_EOS>
            # target: V V V V V V <_EOS>
            # input: <_GO> V V V V V V
            one_hot_headline_target = one_hot_headline_raw[1:]
            one_hot_headline_input = one_hot_headline_raw[:-1]

            yield one_hot_article, one_hot_headline_input, one_hot_headline_target, count_article, count_headline
            sentence_article = bytes.decode(file_article.readline())
            sentence_headline = bytes.decode(file_headline.readline())

        file_headline.close()
        file_article.close()


def simple_word_count(file_name):
    filename = './data/' + file_name + '.txt'
    countname = './data/words_count.txt'
    file_count = open(countname, 'w')
    for sentence in open(filename):
        length = len(sentence.split())
        file_count.writelines(str(length))
        file_count.write('\n')
    print('simple_word_count complete')


def main():
    mostnwords = get_words('traintext', 3650)
    # print(mostnwords)
    changetext(32, 'article', mostnwords)
    # simple_word_count('article_train')


if __name__ == '__main__':
    main()

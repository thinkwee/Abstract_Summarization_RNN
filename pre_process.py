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
    most_n_words = set()
    for word, num in word_num:
        most_n_words.add(word)
    return most_n_words


def changetext(batch_size, file_name, most_n_words):
    # add <_GO>,<_EOS>,<_PAD>,<_UNK> to the file_name.txt

    filename = './data/' + file_name + '.txt'

    filename_processed = './data/' + file_name + '_processed.txt'
    file_processed = open(filename_processed, 'w')

    # calculate the max sentence length in each batch
    batch_length = []
    max_length = 0
    index = 0
    for sentence in open(filename):
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

    # add <_GO>,<_EOS>,<_UNK>,<_PAD> to the sentence
    # tf.dynamic_rnn still need padded sentences in each batch
    batch_idx = 0
    index = 0
    if file_name == 'headline' or file_name == "headline_middle":
        for line in open(filename):
            newline = ""
            newline += '_GO '
            for word in line.split():
                if word not in most_n_words:
                    newline += '_UNK '
                else:
                    newline += word + ' '
            newline += '_EOS'
            for _ in range(batch_length[batch_idx] - len(line.split())):
                newline += ' _PAD'
            index += 1
            if index == batch_size:
                batch_idx += 1
                index = 0
            newline += '\n'
            file_processed.writelines(newline)
        print('pre processing headline finished')
    elif file_name == 'article' or file_name == 'article_middle':
        for line in open(filename):
            newline = ""
            for word in line.split():
                if word not in most_n_words:
                    newline += '_UNK '
                else:
                    newline += word + ' '
            for _ in range(batch_length[batch_idx] - len(line.split())):
                newline += '_PAD '
            index += 1
            if index == batch_size:
                batch_idx += 1
                index = 0
            newline += '\n'
            file_processed.writelines(newline)
        print('pre processing article finished')
    else:
        print('wrong during processing,please verify your file name')
    file_processed.close()


def get_batch(batch_size, iterator):
    # batch generalization
    while True:
        encoder_batch = []
        decoder_batch = []
        target_batch = []
        encoder_length_batch = []
        decoder_length_batch = []

        for index in range(batch_size):
            encoder_input_single, decoder_input_single, target_single, encoder_length_single_real, decoder_length_single_real = next(
                iterator)

            encoder_batch.append(encoder_input_single)
            decoder_batch.append(decoder_input_single)
            target_batch.append(target_single)
            encoder_length_batch.append(encoder_length_single_real)
            decoder_length_batch.append(decoder_length_single_real)

        decoder_max_iter = np.max(decoder_length_batch)

        yield encoder_batch, decoder_batch, target_batch, encoder_length_batch, decoder_length_batch, decoder_max_iter


def one_hot_generate(one_hot_dictionary, epoch, is_train):
    # generate each feed data

    for i in range(epoch):
        if is_train:
            file_headline = open('./data/headline_middle_train.txt', 'rb')
            file_article = open('./data/article_middle_train.txt', 'rb')
        else:
            file_headline = open('./data/headline_middle_test.txt', 'rb')
            file_article = open('./data/article_middle_test.txt', 'rb')

        sentence_article = bytes.decode(file_article.readline())
        sentence_headline = bytes.decode(file_headline.readline())

        while sentence_article and sentence_headline:
            words_article = []
            words_headline = []
            count_headline = 0
            count_article = 0
            count_article_real = 0
            count_headline_real = 0
            for word in sentence_article.split():
                word = word.strip()
                words_article.append(word)
                if word != '_PAD':
                    count_article_real += 1
                count_article += 1
            for word in sentence_headline.split():
                word = word.strip()
                words_headline.append(word)
                if word != '_PAD':
                    count_headline_real += 1
                count_headline += 1
            one_hot_article = np.zeros([count_article], dtype=int)
            one_hot_headline_raw = np.zeros([count_headline], dtype=int)

            # one_hot_dictionary['_UNK']=1
            # TODO:should look up the _UNK id not appoint 1
            for index, word in enumerate(words_article):
                one_hot_article[index] = one_hot_dictionary[word] if word in one_hot_dictionary else 1

            for index, word in enumerate(words_headline):
                one_hot_headline_raw[index] = one_hot_dictionary[word] if word in one_hot_dictionary else 1

            # raw: <_GO> V1 V2 V3 V4 V5 V6 <_EOS>
            # target: V1 V2 V3 V4 V5 V6 <_EOS>
            # input: <_GO> V1 V2 V3 V4 V5 V6
            one_hot_headline_input = one_hot_headline_raw[:-1]
            one_hot_headline_target = one_hot_headline_raw[1:]

            # resize the length
            count_headline_real -= 1

            yield one_hot_article, one_hot_headline_input, one_hot_headline_target, count_article_real, count_headline_real
            sentence_article = bytes.decode(file_article.readline())
            sentence_headline = bytes.decode(file_headline.readline())

        file_headline.close()
        file_article.close()


def simple_word_count(file_name):
    filename = './data/' + file_name + '.txt'
    count_name = './data/words_count.txt'
    file_count = open(count_name, 'w')
    for sentence in open(filename):
        length = len(sentence.split())
        file_count.writelines(str(length))
        file_count.write('\n')
    print('simple_word_count complete')


def main():
    vocab_size = 3000
    batch_size = 32
    most_n_words = get_words('traintext_raw', vocab_size)
    # changetext(batch_size, 'article', most_n_words)
    # changetext(batch_size, 'headline', most_n_words)
    changetext(batch_size, 'article_middle', most_n_words)
    changetext(batch_size, 'headline_middle', most_n_words)
    # simple_word_count('article_train')


if __name__ == '__main__':
    main()

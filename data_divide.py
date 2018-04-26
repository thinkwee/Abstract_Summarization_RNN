import string


def check_count():
    sentence_count = 0
    file_article = open('./data/article_processed.txt', 'rb')
    sentence_article = bytes.decode(file_article.readline())
    file_headline = open('./data/headline_processed.txt', 'rb')
    sentence_headline = bytes.decode(file_headline.readline())
    while sentence_article and sentence_headline:
        sentence_count = sentence_count + 1
        sentence_article = bytes.decode(file_article.readline())
        sentence_headline = bytes.decode(file_headline.readline())
    file_article.close()
    file_headline.close()
    return sentence_count


def divide():
    # divide file into train and test part

    count = 1

    # train:test=19:1
    # TODO:the number should be calculated and be the multiple of batch_size
    # TODO:if the batch size be changed,the seq2seq model will go into wrong due to codes here
    line_test = 6880

    file_article_test = open('./data/article_test.txt', 'w')
    file_headline_test = open('./data/headline_test.txt', 'w')
    file_article_train = open('./data/article_train.txt', 'w')
    file_headline_train = open('./data/headline_train.txt', 'w')
    file_article = open('./data/article_processed.txt', 'rb')
    file_headline = open('./data/headline_processed.txt', 'rb')
    file_w2v_train = open('./data/traintext.txt', 'w')

    sentence_headline = bytes.decode(file_headline.readline())
    sentence_article = bytes.decode(file_article.readline())

    while sentence_article and sentence_headline:
        if count > line_test:
            file_article_test.writelines(sentence_article)
            file_headline_test.writelines(sentence_headline)
        else:
            file_article_train.writelines(sentence_article)
            file_headline_train.writelines(sentence_headline)
        file_w2v_train.writelines(sentence_headline)
        file_w2v_train.writelines(sentence_article)
        sentence_article = bytes.decode(file_article.readline())
        sentence_headline = bytes.decode(file_headline.readline())
        count += 1

    file_article.close()
    file_headline.close()
    file_headline_train.close()
    file_article_train.close()
    file_article_test.close()
    file_headline_test.close()

    print("divide complete")


def statistics(file_path):
    # check the average length and max length of the input and target
    file = open(file_path, 'rb')
    words = {}
    max_sentence_length = 0
    sentence = bytes.decode(file.readline())
    sentence_count = 0
    vocab_count = 0
    words_count = 0
    strip = string.whitespace + string.punctuation + "\"'"
    while sentence:
        sentence_length = 0
        for word in sentence.split():
            word = word.strip(strip)
            words[word] = words.get(word, 0) + 1
            sentence_length += 1
        words_count += sentence_length
        if sentence_length > max_sentence_length:
            max_sentence_length = sentence_length
        sentence_count += 1
        sentence = bytes.decode(file.readline())
    for word in sorted(words):
        print("'{0}' occurs {1} times".format(word, words[word]))
        vocab_count += 1
    print("total {0} sentences".format(sentence_count))
    print("total {0} vocab".format(vocab_count))
    print("total {0} words".format(words_count))
    print("max sentence length: {0}".format(max_sentence_length))
    print("average sentence length: {0}".format(words_count / sentence_count))


def main():
    statistics("./data/negative.txt")
    # print(check_count())
    # divide()


if __name__ == '__main__':
    main()
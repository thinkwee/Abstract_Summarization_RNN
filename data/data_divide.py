import string


# train:test=19:1

def init():
    sentence_count = 0
    file_article = open('./article.txt', 'rb')
    sentence_article = bytes.decode(file_article.readline())
    file_headline = open('./headline.txt', 'rb')
    sentence_headline = bytes.decode(file_headline.readline())
    while sentence_article and sentence_headline:
        sentence_count = sentence_count + 1
        sentence_article = bytes.decode(file_article.readline())
        sentence_headline = bytes.decode(file_headline.readline())
    file_article.close()
    file_headline.close()
    return sentence_count


def divide(sentence_count):
    count = 1
    # line_validate = sentence_count / 20 * 18
    line_test = sentence_count / 20 * 19
    # file_article_validate = open('./article_validate.txt', 'w')
    # file_headline_validate = open('./headline_validate.txt', 'w')
    file_article_test = open('./article_test.txt', 'w')
    file_headline_test = open('./headline_test.txt', 'w')
    file_article_train = open('./article_train.txt', 'w')
    file_headline_train = open('./headline_train.txt', 'w')
    file_article = open('./article.txt', 'rb')
    file_headline = open('./headline.txt', 'rb')
    sentence_headline = bytes.decode(file_headline.readline())
    sentence_article = bytes.decode(file_article.readline())

    while sentence_article and sentence_headline:
        if count > line_test:
            file_article_test.writelines(sentence_article)
            file_headline_test.writelines(sentence_headline)
        # elif count > line_validate:
        #     file_article_validate.writelines(sentence_article)
        #     file_headline_validate.writelines(sentence_headline)
        else:
            file_article_train.writelines(sentence_article)
            file_headline_train.writelines(sentence_headline)
        sentence_article = bytes.decode(file_article.readline())
        sentence_headline = bytes.decode(file_headline.readline())
        count += 1
    file_article.close()
    file_headline.close()
    file_headline_train.close()
    file_article_train.close()
    # file_article_validate.close()
    # file_headline_validate.close()
    file_article_test.close()
    file_headline_test.close()


def statistics(file_path):
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
    statistics("./traintext.txt")
    # sentence_count = init()


if __name__ == '__main__':
    main()

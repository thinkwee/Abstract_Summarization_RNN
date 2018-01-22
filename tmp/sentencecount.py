import string

file_article = open('./article.txt', 'rb')
sentence_article = bytes.decode(file_article.readline())
file_headline = open('./headline.txt', 'rb')
sentence_headline = bytes.decode(file_headline.readline())
sentence_count = 0
max_article = 0
max_headline = 0
total_article = 0
total_headline = 0

while sentence_article and sentence_headline:
    sentence_count = sentence_count + 1
    words_article = []
    words_headline = []
    sentence_article = bytes.decode(file_article.readline())
    sentence_headline = bytes.decode(file_headline.readline())
    strip = string.whitespace + string.punctuation + "\"'"
    for word in sentence_article.split():
        word = word.strip(strip)
        words_article.append(word)
    for word in sentence_headline.split():
        word = word.strip(strip)
        words_headline.append(word)
    if len(words_headline) > max_headline:
        max_headline = len(words_headline)
    if len(words_article) > max_article:
        max_article = len(words_article)
    total_article += len(words_article)
    total_headline += len(words_headline)

print(sentence_count)
print(max_headline)
print(max_article)
print(total_article/sentence_count)
print(total_headline/sentence_count)

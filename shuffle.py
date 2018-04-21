import numpy as np


def shuffle_train_data():
    lines_headline = []
    lines_article = []

    with open("./data/headline_train.txt", 'r') as infile:
        for line in infile:
            lines_headline.append(line)
    infile.close()
    with open("./data/article_train.txt", 'r') as infile:
        for line in infile:
            lines_article.append(line)
    infile.close()

    index = np.arange(6880)
    np.random.shuffle(index)

    out_headline = open("./data/headline_train.txt", 'w')
    out_article = open("./data/article_train.txt", 'w')

    for i in range(6880):
        out_article.writelines(lines_article[index[i]])
        out_headline.writelines(lines_headline[index[i]])

    out_article.close()
    out_headline.close()


shuffle_train_data()

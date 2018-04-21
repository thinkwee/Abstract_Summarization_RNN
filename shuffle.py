import numpy as np


# batch_size=32
# 6880/32=215
# 215-1=214


def shuffle_train_data():
    lines_headline = []
    lines_article = []
    lines_headline_batch = []
    lines_article_batch = []

    index = 0
    with open("./data/headline_train.txt", 'r') as infile:
        for line in infile:
            lines_headline_batch.append(line)
            index += 1
            if index % 32 == 0:
                lines_headline.append(lines_headline_batch)
                lines_headline_batch = []
    infile.close()

    index = 0
    with open("./data/article_train.txt", 'r') as infile:
        for line in infile:
            lines_article_batch.append(line)
            index += 1
            if index % 32 == 0:
                lines_article.append(lines_article_batch)
                lines_article_batch = []
    infile.close()

    index = np.arange(214)
    np.random.shuffle(index)

    out_headline = open("./data/headline_train.txt", 'w')
    out_article = open("./data/article_train.txt", 'w')

    for i in range(214):
        for line in lines_article[index[i]]:
            out_article.writelines(line)
        for line in lines_headline[index[i]]:
            out_headline.writelines(line)

    for line in lines_article[214]:
        out_article.writelines(line)

    for line in lines_headline[214]:
        out_headline.writelines(line)
    out_article.close()
    out_headline.close()


shuffle_train_data()

import numpy as np


# batch_size=32
# 6880/32=215
# 215-1=214
# for the middle copora the size is 99840/32=3120-1=3119


def shuffle_train_data():
    lines_headline = []
    lines_article = []
    lines_headline_batch = []
    lines_article_batch = []

    index = 0
    with open("./data/headline_middle_train.txt", 'r') as infile:
        for line in infile:
            lines_headline_batch.append(line)
            index += 1
            if index % 32 == 0:
                lines_headline.append(lines_headline_batch)
                lines_headline_batch = []
    infile.close()

    index = 0
    with open("./data/article_middle_train.txt", 'r') as infile:
        for line in infile:
            lines_article_batch.append(line)
            index += 1
            if index % 32 == 0:
                lines_article.append(lines_article_batch)
                lines_article_batch = []
    infile.close()

    index = np.arange(3119)
    np.random.shuffle(index)

    out_headline = open("./data/headline_middle_train.txt", 'w')
    out_article = open("./data/article_middle_train.txt", 'w')

    for i in range(3119):
        for line in lines_article[index[i]]:
            out_article.writelines(line)
        for line in lines_headline[index[i]]:
            out_headline.writelines(line)

    # just write the validate data
    for line in lines_article[3119]:
        out_article.writelines(line)

    for line in lines_headline[3119]:
        out_headline.writelines(line)
    out_article.close()
    out_headline.close()


shuffle_train_data()

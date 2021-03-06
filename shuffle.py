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


def shuffle_senti_data():
    # depend on the copora to train
    # little:6880(32*215)
    # mid:99840(32*3120)
    # mid_sen:59200(32*1850)
    # midplus_sen:176000(32*5500)
    # midplus_sen_dedup:153600(32*4800)
    size = 5499
    file_name_headline = "./data/headline_middle_train.txt"
    file_name_article = "./data/article_middle_train.txt"
    # file_name_senti = "./data/middle_sen.txt"
    batch_size = 32

    lines_headline = []
    lines_article = []
    lines_headline_batch = []
    lines_article_batch = []

    index = 0
    with open(file_name_headline, 'r') as infile:
        for line in infile:
            lines_headline_batch.append(line)
            index += 1
            if index % batch_size == 0:
                lines_headline.append(lines_headline_batch)
                lines_headline_batch = []
    infile.close()

    index = 0
    with open(file_name_article, 'r') as infile:
        for line in infile:
            lines_article_batch.append(line)
            index += 1
            if index % batch_size == 0:
                lines_article.append(lines_article_batch)
                lines_article_batch = []
    infile.close()

    # senti_label = open(file_name_senti, 'r').read().split()
    # senti_label_batch = []
    # for i in range(0, len(senti_label), batch_size):
    #     senti_label_batch.append(senti_label[i:i + batch_size])

    index = np.arange(size)
    np.random.shuffle(index)

    out_headline = open(file_name_headline, 'w')
    out_article = open(file_name_article, 'w')
    # out_senti = open(file_name_senti, 'w')

    for i in range(size):
        for line in lines_article[index[i]]:
            out_article.writelines(line)
        for line in lines_headline[index[i]]:
            out_headline.writelines(line)
        # for num in senti_label_batch[index[i]]:
        #     out_senti.write(num + ' ')

    # just write the validate data
    for line in lines_article[size]:
        out_article.writelines(line)

    for line in lines_headline[size]:
        out_headline.writelines(line)

    # for num in senti_label_batch[size]:
    #     out_senti.write(num + ' ')

    out_article.close()
    out_headline.close()
    # out_senti.close()

# shuffle_train_data()

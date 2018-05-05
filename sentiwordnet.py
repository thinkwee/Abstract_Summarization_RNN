from __future__ import division
import nltk


# 1.      CC      Coordinating conjunction
# 2.     CD     Cardinal number
# 3.     DT     Determiner
# 4.     EX     Existential there
# 5.     FW     Foreign word
# 6.     IN     Preposition or subordinating conjunction
# 7.     JJ     Adjective
# 8.     JJR     Adjective, comparative
# 9.     JJS     Adjective, superlative
# 10.     LS     List item marker
# 11.     MD     Modal
# 12.     NN     Noun, singular or mass
# 13.     NNS     Noun, plural
# 14.     NNP     Proper noun, singular
# 15.     NNPS     Proper noun, plural
# 16.     PDT     Predeterminer
# 17.     POS     Possessive ending
# 18.     PRP     Personal pronoun
# 19.     PRP$     Possessive pronoun
# 20.     RB     Adverb
# 21.     RBR     Adverb, comparative
# 22.     RBS     Adverb, superlative
# 23.     RP     Particle
# 24.     SYM     Symbol
# 25.     TO     to
# 26.     UH     Interjection
# 27.     VB     Verb, base form
# 28.     VBD     Verb, past tense
# 29.     VBG     Verb, gerund or present participle
# 30.     VBN     Verb, past participle
# 31.     VBP     Verb, non-3rd person singular present
# 32.     VBZ     Verb, 3rd person singular present
# 33.     WDT     Wh-determiner
# 34.     WP     Wh-pronoun
# 35.     WP$     Possessive wh-pronoun
# 36.     WRB     Wh-adverb


class SentiWordNet():
    def __init__(self, netpath):
        self.netpath = netpath
        self.dictionary = {}

    def infoextract(self):
        tempdict = {}
        f = open(self.netpath, "r")
        # Example line:
        # POS     ID     PosS  NegS SynsetTerm#sensenumber Desc
        # a   00009618  0.5    0.25  spartan#4 austere#3 ascetical#2  ……
        for sor in f.readlines():
            if sor.strip().startswith("#"):
                pass
            else:
                data = sor.split("\t")
                if len(data) != 6:
                    print('invalid data')
                    break
                word_type_marker = data[0]
                synset_score = float(data[2]) - float(data[3])  # // Calculate synset score as score = PosS - NegS
                syn_terms_split = data[4].split(" ")  # word#sentimentscore
                for w in syn_terms_split:
                    syn_term_and_rank = w.split("#")
                    syn_term = syn_term_and_rank[0] + "#" + word_type_marker  # 单词#词性
                    syn_term_rank = int(syn_term_and_rank[1])
                    if syn_term in tempdict:
                        t = [syn_term_rank, synset_score]
                        tempdict.get(syn_term).append(t)
                    else:
                        temp = {syn_term: []}
                        t = [syn_term_rank, synset_score]
                        temp.get(syn_term).append(t)
                        tempdict.update(temp)

        for key in tempdict.keys():
            score = 0.0
            ssum = 0.0
            for wordlist in tempdict.get(key):
                score += wordlist[1] / wordlist[0]
                ssum += 1.0 / wordlist[0]
                score /= ssum
                self.dictionary.update({key: score})

    def getscore(self, word, pos):
        return self.dictionary.get(word + "#" + pos)


def make_np_vector(np_dict, pos_info):
    # 名词（n）、形容词（a）、动词（v）和副词（r）
    # 返回向量(正向情感平均值，负向情感平均值，名词占比，形容词占比，动词占比，副词占比)
    count_total = len(pos_info)

    n_total = 0
    a_total = 0
    v_total = 0
    r_total = 0

    pos_total = 0.0
    neg_total = 0.0
    pos_count = 0
    neg_count = 0

    noun_set = {'NN', 'NNS', 'NNP', 'NNPS'}
    adj_set = {'JJ', 'JJR', 'JJS'}
    verb_set = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    adv_set = {'RB', 'RBR', 'RBS', 'WRB'}

    vector = []

    for i in range(count_total):
        pos = 'n'
        if pos_info[i][1] in noun_set:
            pos = 'n'
            n_total += 1
        elif pos_info[i][1] in adj_set:
            pos = 'a'
            a_total += 1
        elif pos_info[i][1] in verb_set:
            pos = 'v'
            v_total += 1
        elif pos_info[i][1] in adv_set:
            pos = 'r'
            r_total += 1

        if np_dict.getscore(pos_info[i][0], pos) is not None:
            count = np_dict.getscore(pos_info[i][0], pos)
            if count > 0:
                pos_total += count
                pos_count += 1
            elif count < 0:
                neg_total += count
                neg_count += 1
    if pos_count > 0:
        vector.append(pos_total / pos_count)
    else:
        vector.append(0)

    if neg_count > 0:
        vector.append(neg_total / neg_count)
    else:
        vector.append(0)

    if count_total > 0:
        vector.append(n_total / count_total)
        vector.append(a_total / count_total)
        vector.append(v_total / count_total)
        vector.append(r_total / count_total)
    else:
        for _ in range(4):
            vector.append(0)

    return vector


def test():
    net_path = "./data/SentiWordNet.txt"
    swn = SentiWordNet(net_path)
    swn.infoextract()
    sentence = "Don't buy this unless you are willing to pay twice as much for the product you need. Turbo Tax has " \
               "lowered themselves to deceptive advertizing as far as I see it. "
    text = nltk.word_tokenize(sentence)
    pos_info = nltk.pos_tag(text)
    return make_np_vector(swn, pos_info)


if __name__ == '__main__':
    test()

    # print("good#a " + str(swn.getscore('good', 'a')))
    # print("good#n " + str(swn.getscore('good', 'n')))
    # print("good#r " + str(swn.getscore('good', 'r')))
    # print("good#v " + str(swn.getscore('good', 'v')))
    #
    # print("bad#a " + str(swn.getscore('bad', 'a')))
    # print("bad#n " + str(swn.getscore('bad', 'n')))
    # print("bad#r " + str(swn.getscore('bad', 'r')))
    # print("bad#v " + str(swn.getscore('bad', 'v')))
    #
    # print("happy#a " + str(swn.getscore('happy', 'a')))
    # print("happy#n " + str(swn.getscore('happy', 'n')))
    # print("happy#r " + str(swn.getscore('happy', 'r')))
    # print("happy#v " + str(swn.getscore('happy', 'v')))

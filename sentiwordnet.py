from __future__ import division


class SentiWordNet():
    def __init__(self, netpath):
        self.netpath = netpath
        self.dictionary = {}

    def infoextract(self):
        tempdict = {}
        f = open(self.netpath, "r")
        print('start extracting.......')
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

    def getscore(self, word):
        return self.dictionary.get(word + "#a"), self.dictionary.get(word + "#n"), self.dictionary.get(
            word + "#r"), self.dictionary.get(word + "#v")


if __name__ == '__main__':
    netpath = "./data/SentiWordNet.txt"
    swn = SentiWordNet(netpath)
    swn.infoextract()

    testsentece = "australian shares closed down  percent monday following a weak lead from the united states and lower commodity prices  dealers said"
    sentence = testsentece.split()

    for word in sentence:
        count = 0.0
        a, n, r, v = swn.getscore(word)
        print(word, a, n, r, v)

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

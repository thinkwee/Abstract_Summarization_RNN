import random

file = open("./unk_test.txt", "r")
file_headline = open("./unk_test_headline.txt", "r")
headline = file_headline.read()
headline = headline.split()
# print(headline)
count = 0
for line in file:
    print(count, end=' ')
    count += 1
    s = line.split()
    l = len(s)
    for index in range(30):
        i = random.randint(0, l - 1)
        if s[i] in headline and index % 2 == 0:
            print(s[i], end=' ')
        elif index % 2 != 0:
            print(s[i], end=' ')
    print()

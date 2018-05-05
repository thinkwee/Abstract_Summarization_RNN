file_output = open('./temp.txt', 'w')
for i in range(64):
    sentence = open('./test' + str(i + 1) + '.txt', 'r').read()
    # file_output.write(str(i) + '  ')
    file_output.write(sentence)
    file_output.write('\n')
file_output.close()

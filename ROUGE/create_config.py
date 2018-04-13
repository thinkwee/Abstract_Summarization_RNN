file = open("./config.txt", 'w')
for i in range(100):
    file.writelines("/home/cmy/ROUGE-1.5.5/RELEASE-1.5.5/lw1/systems/test" + str(
        i) + ".txt /home/cmy/ROUGE-1.5.5/RELEASE-1.5.5/lw1/models/test" + str(i) + ".txt\n")
file.close()

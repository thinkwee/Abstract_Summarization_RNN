file = open("./config.txt", 'w')
for i in range(100):
    file.writelines("/home/cmy/ROUGE-1.5.5/RELEASE-1.5.5/lw/systems/test" + str(
        i) + ".txt /home/cmy/ROUGE-1.5.5/RELEASE-1.5.5/lw/models/test" + str(i) + ".txt\n")
file.close()

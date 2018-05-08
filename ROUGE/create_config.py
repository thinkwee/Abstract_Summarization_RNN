file = open("./config.txt", 'w')
for i in range(32):
    file.writelines("/home/cmy/ROUGE-1.5.5/RELEASE-1.5.5/lw_nosen/systems/test" + str(
        i) + ".txt /home/cmy/ROUGE-1.5.5/RELEASE-1.5.5/lw_nosen/models/test" + str(i) + ".txt\n")
file.close()

def make_file_model():
    with open('headline_test_100.txt') as f:
        count = 0
        for line in f:
            file_create = open("./systems/test" + str(count) + ".txt", "w")
            file_create.writelines(line)
            count += 1
            file_create.close()
            if count == 32:
                break
    print("make models files complete")


def main():
    make_file_model()


if __name__ == '__main__':
    main()

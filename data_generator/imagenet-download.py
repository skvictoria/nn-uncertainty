with open('/home/seulgi/work/nn-uncertainty/data_generator/wordnet.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        id_ = line.split()[0]  # Extract the first part
        command = f"wget https://image-net.org/data/winter21_whole/{id_}.tar -O /home/seulgi/work/data/imagenet/{id_}.tar && " \
                  f"mkdir -p /home/seulgi/work/data/imagenet/{id_} && " \
                  f"tar -xf /home/seulgi/work/data/imagenet/{id_}.tar -C /home/seulgi/work/data/imagenet/{id_} && " \
                  f"rm /home/seulgi/work/data/imagenet/{id_}.tar"
        print(command)

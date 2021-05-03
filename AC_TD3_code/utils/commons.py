import pickle

def load_data(path):
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

def save_data(path, data):
    file = open(path, "wb")
    pickle.dump(data, file)
    file.close()

def chunker_list(seq, size):
    return [seq[pos: pos + size] for pos in range(0, len(seq), size)]

def chunker_num(num, size):
    return [list(range(num))[pos: pos + size] for pos in range(0, num, size)]
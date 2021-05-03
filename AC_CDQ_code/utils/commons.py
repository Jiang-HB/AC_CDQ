import pickle

def save_data(data, filename):
    file = open(filename, "wb")
    pickle.dump(data, file)
    file.close()

def load_data(filename):
    file = open(filename, "rb")
    data = pickle.load(file)
    file.close()
    return data

def chunker_num(num, size):
    return [list(range(num))[pos: pos + size] for pos in range(0, num, size)]

def chunker_list(seq, size):
    return [seq[pos: pos + size] for pos in range(0, len(seq), size)]
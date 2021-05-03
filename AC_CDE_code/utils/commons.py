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
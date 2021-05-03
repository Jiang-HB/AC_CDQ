import os
from utils import save_data

class Recoder:
    def __init__(self, results_dir, seed):
        self.train_returns, self.test_returns = [], []
        self.train_returns_path = os.path.join(results_dir, "train_returns_seed%d.pth" % (seed))
        self.test_returns_path = os.path.join(results_dir, "test_returns_seed%d.pth" % (seed))

    def add_result(self, result, result_type):
        if result_type == "train_return":
            self.train_returns.append(result)
        if result_type == "test_return":
            self.test_returns.append(result)

    def save_result(self):
        save_data(self.train_returns_path, self.train_returns)
        save_data(self.test_returns_path, self.test_returns)



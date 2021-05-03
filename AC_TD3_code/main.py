import os
from utils import opts, run

def main():

    if not os.path.exists(opts.results_dir):
        os.mkdir(opts.results_dir)

    run(opts, opts.seed)

if __name__ == '__main__':
    main()
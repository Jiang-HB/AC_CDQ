import numpy as np, pdb
from multiprocessing import Pipe

def play_mp(opts, pool, args, fn):
    pipes = [Pipe() for _ in range(opts.n_core)]
    rewards = np.zeros((opts.n_repeat, opts.n_step))
    max_Q0 = np.zeros((opts.n_repeat, opts.n_step))

    for i in range(0, len(args), opts.n_core):
        print("start: %d" % i)
        args_batch = args[i: i + opts.n_core]
        ps = [pool.apply_async(fn, args=[opts] + arg + [pipes[i][1]]) for i, arg in enumerate(args_batch)]
        [p.get() for p in ps]

        for pipe in pipes[:len(args_batch)]:
            repeat_n, _rewards, _max_Q = pipe[0].recv()
            rewards[repeat_n] = _rewards
            max_Q0[repeat_n] = _max_Q
            # print("end %d" % repeat_n)

    return rewards, max_Q0

# def play_mp(opts, pool, args, fn):
#     rewards = np.zeros((opts.n_repeat, opts.n_step))
#     max_Q0 = np.zeros((opts.n_repeat, opts.n_step))
#
#     def callback_fn(x):
#         repeat_n, _rewards, _max_Q0 = x
#         rewards[repeat_n] = _rewards
#         max_Q0[repeat_n] = _max_Q0
#
#     rcd = 0
#     for args_batch in chunker_list(args, opts.n_core):
#         print("rcd %d" % (rcd))
#         ps = [pool.apply_async(fn, args=[opts] + arg, callback=callback_fn) for arg in args_batch]
#         [p.get() for p in ps]
#         rcd += len(args_batch)
#
#     return rewards, max_Q0

# def play_mp(opts, pool, args, fn):
#     rewards = np.zeros((opts.n_repeat, opts.n_step))
#     max_Q0 = np.zeros((opts.n_repeat, opts.n_step))
#
#     def callback_fn(x):
#         repeat_n, _rewards, _max_Q0 = x
#         rewards[repeat_n] = _rewards
#         max_Q0[repeat_n] = _max_Q0
#
#     rcd = 0
#     pipes = [Pipe() for _ in range(opts.n_core)]
#     for args_batch in chunker_list(args, opts.n_core):
#         print("rcd %d" % (rcd))
#         ps = [pool.apply_async(fn, args=[opts, pipes[i][1]] + arg) for i, arg in enumerate(args_batch)]
#         [p.get() for p in ps]
#         rcd += len(args_batch)
#
#         for pipe in pipes[: len(args_batch)]:
#             callback_fn(pipe[0].recv())
#
#     return rewards, max_Q0
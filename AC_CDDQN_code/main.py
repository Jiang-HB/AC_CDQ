import torch, os
from utils import opts
from algorithms.ac_cddqn import AC_CDDQN as Method

def main():

    opts.device = torch.device("cuda")
    print('Cuda available?: ' + str(torch.cuda.is_available()))

    for id in opts.ids:

        opts.id = id
        opts.tag = "%s_%s_id%d" % (opts.env_nm, opts.alg_tag, opts.id)
        opts.save_dir = "./results/%s/%s" % (opts.agent_nm, opts.tag)
        if not os.path.exists(opts.save_dir):
            os.mkdir(opts.save_dir)

        agent = Method(opts)
        agent.run_steps()

if __name__ == '__main__':
    main()
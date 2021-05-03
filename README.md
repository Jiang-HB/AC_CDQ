# Action Candidate Based Clipped Double Q-learning for Discrete and Continuous Actions

PyTorch implementation of our action candidate based clipped double estimator (AC-CDE), action candidate based clipped Double Q-learning (AC-CDQ), action candidate based clipped Double DQN (AC-CDDQN) and action candidate based TD3 (AC-TD3).

Paper link [arXiv](https://arxiv.org/submit/3726215/view).

### Usage

1. For AC-CDE, we evaluate it on the multi-armed bandits problem. The result can be reproduced by running:

   ```
   cd AC_CDE_code
   python3 main.py
   ```

2. For AC-CDQ, we evaluate it on the grid world game. The result can be reproduced by running:

   ```
   cd AC_CDQ_code
   python3 main.py
   ```

3. For AC-CDDQN, we evaluate it on the MinAtar benchmark. The result can be reproduced by running:

   ```
   cd AC_CDDQN_code
   CUDA_VISIBLE_DEVICES=0 python3 main.py
   ```

4. For AC-TD3, we evaluate it on MuJoCo continuous control tasks. The result can be reproduced by running:

   ```
   cd AC_TD3_code
   CUDA_VISIBLE_DEVICES=0 python3 main.py
   ```

   

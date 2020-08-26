import numpy as np
import sys

trial = 100000
game_num = 100
res_arr = np.zeros(trial)

for i in range(trial):
    win_rate = np.random.binomial(game_num, 0.5) / game_num
    dr = 0.
    if win_rate > 0.5:
        dr = -np.log(1. / win_rate - 1)
    res_arr[i] = dr
    #print(win_rate, dr)
    sys.stderr.write('\r\033{}/{}'.format(i + 1, trial))
    sys.stderr.flush()

print("mean={}, std={}".format(np.mean(res_arr), np.std(res_arr)))



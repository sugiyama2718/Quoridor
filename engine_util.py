import os
from CNNAI import CNNAI

def prepare_AI(parameter_path, color, search_nodes, tau, level, seed):
    # level==-1のときは最新epochのものを読み込む
    files = os.listdir(parameter_path)
    files = [x for x in files if x.startswith("epoch")]
    epochs = [int(x.split(".")[0][5:]) for x in files]
    epochs = list(set(epochs))
    epochs = sorted(epochs)
    if level == 0:
        agent = CNNAI(color, search_nodes=search_nodes, tau=tau, seed=seed, p_is_almost_flat=True, all_parameter_zero=True)
    elif level == -1:
        agent = CNNAI(color, search_nodes=search_nodes, tau=tau, seed=seed)
        target_epoch = epochs[-1]
        agent.load(os.path.join(parameter_path, f"epoch{target_epoch}.ckpt"))
    else:
        agent = CNNAI(color, search_nodes=search_nodes, tau=tau, seed=seed)
        target_epoch = epochs[level - 1]
        agent.load(os.path.join(parameter_path, f"epoch{target_epoch}.ckpt"))
    return agent

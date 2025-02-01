import os
from CNNAI import CNNAI

def prepare_AI(parameter_path, color, search_nodes, tau, level, seed, p_tau=1.0, post_alpha=1.0, post_beta=1.0, C_puct=2.0, opening_tree_path=None):
    # level==-1のときは最新epochのものを読み込む
    files = os.listdir(parameter_path)
    files = [x for x in files if x.startswith("epoch")]
    epochs = [int(x.split(".")[0][5:]) for x in files]
    epochs = list(set(epochs))
    epochs = sorted(epochs)
    if level == 0:
        agent = CNNAI(color, search_nodes=search_nodes, tau=tau, seed=seed, p_is_almost_flat=True, all_parameter_zero=True, p_tau=p_tau, post_alpha=post_alpha, post_beta=post_beta, C_puct=C_puct, opening_tree_path=opening_tree_path)
    elif level == -1:
        agent = CNNAI(color, search_nodes=search_nodes, tau=tau, seed=seed, p_tau=p_tau, post_alpha=post_alpha, post_beta=post_beta, C_puct=C_puct, opening_tree_path=opening_tree_path)
        target_epoch = epochs[-1]
        agent.load(os.path.join(parameter_path, f"epoch{target_epoch}.ckpt"))
    else:
        agent = CNNAI(color, search_nodes=search_nodes, tau=tau, seed=seed, p_tau=p_tau, post_alpha=post_alpha, post_beta=post_beta, C_puct=C_puct, opening_tree_path=opening_tree_path)
        target_epoch = epochs[level - 1]
        agent.load(os.path.join(parameter_path, f"epoch{target_epoch}.ckpt"))
    return agent

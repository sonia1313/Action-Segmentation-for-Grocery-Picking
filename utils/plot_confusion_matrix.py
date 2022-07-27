from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def _plot_cm(cm, path = None):
    # label_to_index_map = {'move-in': 0, 'manipulate': 1, 'grasp': 2, 'pick-up': 3, 'move-out': 4, 'drop': 5}

    y = x = ['move-in', 'manipulate', 'grasp', 'pick-up', 'move-out', 'drop']

    plt.figure(figsize=(10, 10))
    df_cm = pd.DataFrame(cm.cpu().numpy(), index=range(6), columns=range(6))
    sns.set_theme()

    cm_plot = sns.heatmap(df_cm, annot=True, xticklabels=x, yticklabels=y, cbar_kws=None, fmt='g')

    cm_fig = cm_plot.get_figure()


    cm_fig.savefig(path, dpi=cm_fig.dpi)

    plt.close()

    return cm_fig
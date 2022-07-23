from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def _plot_cm(cm, path = None):
    # label_to_index_map = {'move-in': 0, 'manipulate': 1, 'grasp': 2, 'pick-up': 3, 'move-out': 4, 'drop': 5}

    y = x = ['move-in', 'manipulate', 'grasp', 'pick-up', 'move-out', 'drop']

    plt.figure(figsize=(10, 10))
    df_cm = pd.DataFrame(cm.cpu().numpy(), index=range(6), columns=range(6))
    sns.set_theme()

    cm_fig = sns.heatmap(df_cm, annot=True, xticklabels=x, yticklabels=y, cbar_kws=None).get_figure()

    if path:
        cm_fig.savefig(path, dpi = cm_fig.dpi)

    else:
        
        cm_fig.savefig(f"confusion_matrix_figs/{self.experiment_name}_cm_{self.counter}.png", dpi=cm_fig.dpi)

    plt.close(cm_fig)

    return cm_fig
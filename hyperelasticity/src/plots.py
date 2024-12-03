import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
from itertools import zip_longest

from .cps_colors import CPS_COLORS


def plot_stress_predictions(
        labels_dict: dict[str, np.ndarray], 
        predictions_dict: dict[str, np.ndarray] | None = None, 
        figsize: tuple[int, int] = (10, 8),
        colors: list[tuple] = CPS_COLORS,
    ) -> plt.Figure:

    fig, axs = plt.subplots(
        3, 
        3, 
        sharex='col', 
        # sharey='row', 
        figsize=figsize
    )

    lines = []
    names = []

    for (name, labels), color in zip(labels_dict.items(), colors):
        if predictions_dict is not None and name not in predictions_dict.keys():
            raise KeyError(f'key {name} not in prediction dict!')
        
        predictions = predictions_dict[name] if predictions_dict is not None else np.array([])

        # iterate over label values
        for idx, (label_i, preds_i) in enumerate(zip_longest(labels.T, predictions.T, fillvalue=None)):
            ax1_idx = idx // 3
            ax2_idx = idx % 3
            ax: plt.Axes = axs[ax2_idx, ax1_idx]
            true_line, = ax.plot(label_i, '.', lw=2, color=color, markevery=10)
            
            if preds_i is not None:
                pred_line, = ax.plot(preds_i, '-', lw=2, color=color)

            if ax1_idx == 2 and ax2_idx == 0:
                ax.legend()
            
            if ax2_idx == 2:
                ax.set_xlabel(f'$t \, [s]$')

            ax.set_ylabel(f'$P_{{{ax1_idx+1},{ax2_idx+1}}} \, [MP]$')

        lines.append(true_line)
        names.append(f'{name} Ground Truth')
        if predictions_dict is not None:
            lines.append(pred_line)
            names.append(f'{name} Prediction')
            


    fig.legend(lines, names, ncol=4, labelspacing=0.5, bbox_to_anchor=(0.5, 0.), loc='upper center', frameon=False)
    fig.tight_layout()
    plt.show()


def plot_energy_prediction(
        labels_dict: dict[str, np.ndarray], 
        predictions_dict: dict[str, np.ndarray], 
        figsize: tuple[int, int] = (5, 3),
        colors: list[tuple] = CPS_COLORS,
    ) -> plt.Figure:

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    lines = []
    names = []

    for (name, labels), color in zip(labels_dict.items(), colors):
        if name not in predictions_dict.keys():
            raise KeyError(f'key {name} not in prediction dict!')
        
        true_line, = ax.plot(labels, '.', lw=2, color=color, markevery=10)
        pred_line, = ax.plot(predictions_dict[name], '-', lw=2, color=color)

        lines.extend([true_line, pred_line])
        names.extend([f'{name} Ground Truth', f'{name} Prediction'])

    ax.set_ylabel(r'$W \, [\frac{N}{mm^2}]$')
    ax.set_xlabel(r'$t \, [s]$')
    ax.legend()

    fig.legend(lines, names, ncol=2, labelspacing=0.5, bbox_to_anchor=(0.5, 0.), loc='upper center', frameon=False)
    fig.tight_layout()
    plt.show()




def plot_loss(
        loss: np.ndarray, 
        val_loss: np.ndarray | None = None, 
        figsize: tuple[int, int] = (7, 4),
    ) -> plt.Figure:

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    line, = ax.plot(loss, lw=1, label='Training')

    if val_loss is not None:   
        line, = ax.plot(val_loss, lw=1, label='Validation')
        ax.legend()

    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')

    fig.tight_layout()
    plt.show()



def plot_heatmap(
        score_values: np.ndarray,
        cbar_label: str, 
        vmin=None,
        vmax=None,
        title=None,
        **subplot_kwargs
    ) -> plt.Figure:

    fig, ax = plt.subplots(**subplot_kwargs)
    cax = ax.matshow(score_values, cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.colorbar(cax, ax=ax, label=cbar_label)
    ax.set_title('$P_{i,j} [N]$')
    ax.set_ylabel('$i$')
    ax.set_xlabel('$j$')
    ax.grid(False)

    ax.set_xticks(range(score_values.shape[1])) 
    ax.set_xticklabels([f'${i + 1}$' for i in range(score_values.shape[1])])
    ax.xaxis.set_ticks_position('bottom')
    # ax.tick_params(axis='x', which='major', length=2)

    ax.set_yticks(range(score_values.shape[0]))
    ax.set_yticklabels([f'${i + 1}$' for i in range(score_values.shape[0])])
    ax.yaxis.set_ticks_position('left')
    # ax.tick_params(axis='y', which='major', length=2)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_predictions(
        features: np.ndarray, 
        labels: np.ndarray, 
        predictions: np.ndarray, 
        which_features: list | None = None,
        which_labels: list | None = None,
        figsize: tuple[int, int] = (12, 16),
        feature_label = 'C',
        label_label = 'P',
    ) -> plt.Figure:
    if which_features is None:
        which_features = [f for f in range(features.shape[1])]
    if which_labels is None:
        which_labels = [l for l in range(labels.shape[1])]

    fig, axs = plt.subplots(
        len(which_labels), 
        len(which_features), 
        sharex='col', 
        sharey='row', 
        figsize=figsize
    )

    # iterate over features
    for ax1_idx, i_C in enumerate(which_features):

        # iterate over labels
        for ax2_idx, i_P in enumerate(which_labels):
            ax: plt.Axes = axs[ax2_idx, ax1_idx]
            line, = ax.plot(features[:, i_C], labels[:, i_P], lw=1, label='Training')
            line, = ax.plot(features[:, i_C], predictions[:, i_P], lw=1, label='Prediction')
            true_P_idx = (i_P+1) % 3
            true_C_idx = i_C + 1
            ax.set_title(f'{feature_label} ({true_C_idx}, {true_C_idx}), {label_label} ({true_P_idx}, {true_P_idx})')

            if i_C == 2 and i_P == 0:
                ax.legend()
            
            if ax2_idx == (len(which_labels) - 1):
                ax.set_xlabel(feature_label)
            if ax1_idx == 0:
                ax.set_ylabel(label_label)
    return fig



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

    return fig
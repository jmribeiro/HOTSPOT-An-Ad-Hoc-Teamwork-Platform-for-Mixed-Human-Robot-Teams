import matplotlib.pyplot as plt
import numpy as np

from yaaf.visualization import confidence_interval


def confidence_bar_plot(names, means, N, title, x_label, y_label, show, filename=None, colors=None, confidence=0.95, factor=1.0):
    if isinstance(N, int): N = [N for _ in range(len(names))]
    confidence_intervals = [confidence_interval(means[i], N[i], confidence) for i in range(len(means))]
    fig, ax = plt.subplots()
    x_pos = np.arange(len(names))
    bar_plot = ax.bar(x_pos, means, yerr=confidence_intervals, align='center', alpha=0.5, color="green", ecolor='black', capsize=10)
    if colors is not None:
        for c, color in enumerate(colors):
            bar_plot[c].set_color(color)
    if y_label is not None: ax.set_ylabel(y_label)
    if x_label is not None: ax.set_xlabel(x_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.yaxis.grid(True)
    fig.set_size_inches(tuple(np.array([8, 6]) * factor))
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

def bar_plot(names, values, title, x_label, y_label, show, filename=None, colors=None):
    fig, ax = plt.subplots()
    x_pos = np.arange(len(names))
    bar_plot = ax.bar(x_pos, values, align='center', alpha=0.5, color="green", capsize=10)
    if colors is not None:
        for c, color in enumerate(colors):
            bar_plot[c].set_color(color)
    if y_label is not None: ax.set_ylabel(y_label)
    if x_label is not None: ax.set_xlabel(x_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.yaxis.grid(True)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()



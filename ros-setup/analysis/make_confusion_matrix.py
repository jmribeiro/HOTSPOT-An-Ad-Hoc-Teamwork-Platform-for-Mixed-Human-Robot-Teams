import numpy as np
from tqdm import tqdm

import yaaf

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
heatmaps = [
	"BuGn",
	"copper_r",
	"crest_r",
	"gray_r",
	"Greens",
	"Greys",
	"PRGn",
	"PuBu",
	"PuBuGn",
	"Reds"
]
chosen = "Greens"

def make_confusion_matrix(count_matrix):
    num_nodes = count_matrix.shape[0]
    confusion_matrix = np.zeros_like(count_matrix)
    for n1 in range(num_nodes):
        total = count_matrix[n1].sum()
        for n2 in range(num_nodes+1):
            confusion_matrix[n1, n2] = count_matrix[n1, n2] / total
    return confusion_matrix

def smooth_confusion_matrix(confusion_matrix):
    smooth_matrix = np.zeros_like(confusion_matrix)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            smooth_matrix[i, j] = max(0.03, confusion_matrix[i, j])
        smooth_matrix[i] = yaaf.normalize(smooth_matrix[i])
    return smooth_matrix

def plot_confusion_matrix(confusion_matrix):
    nodes_pt = [
        "porta", "meio da sala", "baxter", "mesa do miguel", "mesa do joao", "falha"
    ]
    nodes = [
        "Door", "Open Space", "Robot Station", "Single Workbench", "Double Workbench", "Fail"
    ]
    df_cm = pd.DataFrame(confusion_matrix, index=[n.replace(' ', '\n') for n in nodes[:-1]], columns=[n.replace(' ', '\n') for n in nodes])
    for heatmap in tqdm(heatmaps):
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=1.0, cmap=heatmap)
        plt.title(
            f"Total of {len(speakers)} speakers ({num_files} files) and {int(count_matrix.sum())} phrases.")
        plt.savefig(f"../resources/heatmaps/{heatmap}.png")
        if heatmap == chosen:
            plt.savefig(f"../resources/plots/Confusion.pdf")
            plt.show()
        plt.close()

if __name__ == '__main__':

    directory = "../resources/count matrices"

    num_nodes = 5
    FAIL_NODE = num_nodes

    count_matrix = np.zeros((num_nodes, num_nodes+1))
    num_files = 0
    speakers = []
    for file in yaaf.files(directory):
        if "count_matrix" in file and ".npy" in file:
            num_files += 1
            name = file.split("count_matrix_")[1].split("_")[0].split(".npy")[0]
            if name not in speakers: speakers.append(name)
            current_count_matrix = np.load(f"{directory}/{file}")
            count_matrix += current_count_matrix

    confusion_matrix = make_confusion_matrix(count_matrix)
    smooth_matrix = smooth_confusion_matrix(confusion_matrix)

    plot_confusion_matrix(confusion_matrix)
    #plot_confusion_matrix(smooth_matrix)

    np.save("../resources/count matrices/confusion_matrix", confusion_matrix)
    np.save("../resources/count matrices/smooth_matrix", smooth_matrix)
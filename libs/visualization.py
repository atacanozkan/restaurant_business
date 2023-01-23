
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, plot_roc_curve
from sklearn.tree import export_graphviz
from scipy.cluster.hierarchy import dendrogram
import random
import pydotplus

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show()

# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

# ROC Curve
def plot_roc(log_model, x, y):
    plot_roc_curve(log_model, x, y)
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.show()

def plot_bar(df, x, y, title, save=None):
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x=x, y=y, data=df)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    if save != None:
        plt.savefig(save)

def plot_line(x_data, y_data, data_label, x_label, y_label, title, save=None):
    if len(data_label) > 1:
        for j in range(len(data_label)):
            plt.plot(x_data[j], y_data[j], label=data_label[j], color=random_color())
    else:
        plt.plot(x_data, y_data, label=data_label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
    if save != None:
        plt.savefig(save)

def random_color():
    r = random.random()
    b = random.random()
    g = random.random()
    rand_color = (r, g, b)
    return rand_color

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

def dendogram_graph(cluster_hierarchy, n_cluster, line=0.5):
    plt.figure(figsize=(10, 5))
    plt.title("Hiyerarşik Kümeleme Dendogramı")
    plt.xlabel("Gözlem Birimleri")
    plt.ylabel("Uzaklıklar")
    dend = dendrogram(cluster_hierarchy,
                      truncate_mode="lastp",
                      p=n_cluster,
                      show_contracted=True,
                      leaf_font_size=10)
    plt.axhline(y=line, color='r', linestyle='--')
    plt.show()


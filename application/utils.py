import string
import random
import os
from matplotlib import pyplot as plt
import seaborn as sns


def generate_radom_string(N=7):
    res = ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))
    return res


def plot_graph(prob_dict, title=None, size=None):
    _ = plt.figure(tight_layout=True, figsize=size)
    # fig.add_axes([0, 0, 1, 1])
    labels = []
    values = []
    for lab, val in prob_dict.items():
        labels.append(lab)
        values.append(val)
    sns.barplot(values, labels)
    plt.xticks(rotation=70)
    saved_img = generate_radom_string()
    img_name = 'saved_plots/' + saved_img + '.png'
    saved_img = 'application/static/' + img_name
    base_path = os.path.join(os.getcwd(), saved_img)
    # saved_img = os.path.join(base_path, saved_img)
    if title:
        plt.title(title)
    plt.savefig(base_path)
    return img_name


def preprocess_img(path):
    pass
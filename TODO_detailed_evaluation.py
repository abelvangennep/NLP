# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def eval_report(filename):

    df_model = pd.read_csv("experiments/base_model/" + filename,sep='\t', header=None)
    df_model.columns = ["word", "label", "output"]
    df_model.dropna(subset=["label"], inplace=True)
    print(classification_report(df_model["label"], df_model["output"]))

    return 0


if __name__ == '__main__':
    for filename in ["random_output.tsv", "majority_output.tsv", "length_output.tsv", "frequency_output.tsv", "model_output.tsv"]:
        eval_report(filename)

    print("hyperparameter tuning:")
    for filename in ["1epoch.tsv","3epoch.tsv", "5epoch.tsv", "10epoch.tsv", "13epoch.tsv", "15epoch.tsv", "20epoch.tsv", "25epoch.tsv"]:
        eval_report(filename)
    plt.plot([1,3,5,10,13,15,20], [0.71, 0.76, 0.83, 0.84, 0.84, 0.84, 0.84], "-")
    plt.title("F1 score by epoch size")
    plt.xticks([1,3,5,10,13,15,20])
    plt.xlabel("# epochs")
    plt.ylabel("F1 score")
    plt.show()
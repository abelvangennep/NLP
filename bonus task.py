
#implement question 8, 12-14 for German


import pandas as pd
import spacy
from wordfreq import word_frequency
import os
import matplotlib.pyplot as plt


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


german_train = pd.read_csv("data/original/german/German_Train.tsv",sep='\t', header=None)
german_dev = pd.read_csv("data/original/german/German_Dev.tsv",sep='\t', header=None)
german_test = pd.read_csv("data/original/german/German_Test.tsv",sep='\t', header=None)


def preprocessing(german, print_info=False):
    german.columns=["id", "sentenece", "start", "end", "target", "native", "non_native", "native_complex", "non_native_complex", "complex_label", "prob"]

    print(german.iloc[:, 1:5])
    print(german["target"])

    nlp = spacy.load("de_core_news_sm")

    target_words = german_train.iloc[:, 4]

    german.loc[: ,"target_doc"] = german.loc[:, "target"].apply(lambda x: nlp(x))
    german.loc[:, "number_of_words"] =german.loc[:, "target_doc"].apply(lambda x: len(x))

    selection_german = german[(german['number_of_words'] == 1) & (german["complex_label"] == 1)]

    selection_german.loc[:, "word_length"] = selection_german.loc[:, "target"].str.len()
    selection_german.loc[:, "word_frequency"] = selection_german.loc[:, "target"].apply(lambda x: word_frequency(str(x), 'de'))

    correlation_len =selection_german["prob"].corr(selection_german["word_length"])
    print("pearson correlation (length/complexity)", correlation_len)

    correlation_freq =selection_german["prob"].corr(selection_german["word_frequency"])
    print("pearson correlation (frequency/complexity)", correlation_freq)

    selection_german.loc[:, "pos_tag"] = selection_german.loc[:, "target_doc"].apply(lambda x: x[0].pos_)

    if print_info ==True:
        selection_german.plot.scatter(x ="pos_tag", y="prob", alpha=0.05)
        plt.ylabel("probabilistic complexity")
        plt.xlabel("pos tags")
        plt.show()

        selection_german.plot.scatter(x ="word_length", y="prob", alpha =0.05)
        plt.ylabel("probabilistic complexity")
        plt.xlabel("word length")
        plt.show()
        selection_german.plot.scatter(x ="word_frequency", y="prob", alpha =0.05)
        plt.ylabel("probabilistic complexity")
        plt.xlabel("word frequency")
        plt.show()


    return_german = german[german['number_of_words'] == 1]
    return_german.loc[:, "word_length"] = return_german.loc[:, "target"].str.len()
    return_german.loc[:, "word_frequency"] = return_german.loc[:, "target"].apply(lambda x: word_frequency(str(x), 'de'))
    return_german[return_german["complex_label" == 1]] = "C"

    return return_german["target", "complex_label", "word_length", "word_frequency"]


train =preprocessing(german_train)
test = preprocessing(german_test)
dev = preprocessing(german_dev)

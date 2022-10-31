# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold


import spacy
import numpy as np
import pandas as pd
from wordfreq import word_frequency
import matplotlib.pyplot as plt
from collections import Counter
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.
def part_B():
    dataframe = pd.read_csv("data/original/english/WikiNews_Train.tsv",sep='\t', header=None)
    dataframe.columns = ["id", "sentenece", "start", "end", "target", "native", "non_native", "native_complex", "non_native_complex", "complex_label", "prob"]

    print(dataframe.iloc[:, 9].value_counts())
    print(dataframe.iloc[:, 10].describe())

    nlp = spacy.load("en_core_web_sm")

    target_words = (dataframe.iloc[:, 4])

    len_word = []

    for word in target_words:
        doc = nlp(word)
        len_word.append(len(doc))

    dataframe['#word'] = len_word
    dataframe["target_doc"] = dataframe.loc[:, "target"].apply(lambda x: nlp(x)[0])

    selection_dataframe = dataframe[(dataframe['#word'] == 1) & (dataframe["complex_label"] == 1)]
    selection_dataframe["len_word"] = selection_dataframe.loc[:, "target"].str.len()

    print(selection_dataframe["len_word"])
    print("Number of instances consisting of more than one token", (dataframe["#word"] > 1).sum())
    print("Maximum number of tokens per instance", max(dataframe["#word"]))
    selection_dataframe["word_frequency"] = selection_dataframe.loc[:, "target"].apply(lambda x: word_frequency(x, 'en'))
    print(selection_dataframe["word_frequency"])

    print("pearson correlation (length/complexity)", selection_dataframe["prob"].corr(selection_dataframe["len_word"]))
    print("pearson correlation (frequency/complexity)", selection_dataframe["prob"].corr(selection_dataframe["word_frequency"]))

    selection_dataframe["pos_tag"] = selection_dataframe.loc[:, "target_doc"].apply(lambda x: x.pos_)
    print(selection_dataframe["pos_tag"])

    selection_dataframe.plot.scatter(x ="pos_tag", y="prob", alpha=0.05)
    plt.ylabel("probabilistic complexity")
    plt.xlabel("pos tags")
    plt.show()

    selection_dataframe.plot.scatter(x ="len_word", y="prob", alpha =0.05)
    plt.ylabel("probabilistic complexity")
    plt.xlabel("word length")
    plt.show()
    selection_dataframe.plot.scatter(x ="word_frequency", y="prob", alpha =0.05)
    plt.ylabel("probabilistic complexity")
    plt.xlabel("word frequency")
    plt.show()




def majority_baseline(train_sentences, train_labels, testinput, testlabels):
    predictions = []

    # TODO: determine the majority class based on the training data
    count_complex= 0
    count_noncomplex = 0

    for label in train_labels:
        count_noncomplex +=label.count("N")
        count_complex +=label.count("C")

    if count_complex> count_noncomplex:
        majority_class ="C"
    elif count_noncomplex > count_complex:
        majority_class ="N"
    else:
        raise Exception("values have the same count")

    print("the majority class is:", majority_class)
    predictions = []
    count_complex= 0
    count_noncomplex = 0

    for i, instance in enumerate(testinput):
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions.append([instance, instance_predictions])
        count_noncomplex += testlabels[i].count("N")
        count_complex += testlabels[i].count("C")

    # TODO: calculate accuracy for the test input
    accuracy = 0
    tot_tokens = 0

    for j, instance in enumerate(testinput):
        tokens =testlabels[j].split(" ")
        instance_true = [tokens[i] for i in range(len(tokens))]

        if "N\n" in instance_true:
            index = instance_true.index("N\n")

        if "C\n" in instance_true:
            index = instance_true.index("C\n")

        instance_true[index] =instance_true[index].replace("\n", "")

        tot_tokens += len(tokens)

        accuracy += sum(np.array(predictions[j][1]) == np.array(instance_true))

    accuracy = accuracy/tot_tokens
    print("fraction majority test: ", count_noncomplex/(count_complex+ count_noncomplex))
    return accuracy, predictions

def random_baseline(train_sentences, train_labels, testinput, testlabels):

    predictions = []
    complex_pred = 0
    noncomplex_pred = 0

    for i, instance in enumerate(testinput):
        tokens = instance.split(" ")
        instance_predictions = []

        for t in tokens:
            choice = np.random.binomial(1, 0.5)

            if choice == 1:
                instance_predictions.append("C")
                complex_pred += 1

            elif choice == 0:
                instance_predictions.append("N")
                noncomplex_pred += 1

        predictions.append([instance, instance_predictions])

    accuracy = 0
    tot_tokens = 0

    for j, instance in enumerate(testinput):
        tokens = testlabels[j].split(" ")
        instance_true = [tokens[i] for i in range(len(tokens))]

        if "N\n" in instance_true:
            index = instance_true.index("N\n")

        if "C\n" in instance_true:
            index = instance_true.index("C\n")

        instance_true[index] = instance_true[index].replace("\n", "")

        tot_tokens += len(tokens)
        accuracy += sum(np.array(predictions[j][1]) == np.array(instance_true))

    accuracy = accuracy / tot_tokens
    print("complex, noncomplex predictions: ", complex_pred, noncomplex_pred)
    return accuracy, predictions

def test_length_baseline(dev_sentences, dev_labels):

    thresholds_tried = []
    thresholds_results = []

    for i in range(1, 14):
        threshold = i
        predictions = []
        accuracy = 0

        for instance in dev_sentences:
            tokens = instance.split(" ")
            length_tokens = np.array([len(t) for t in tokens])

            selection = np.zeros(len(length_tokens), dtype="object")

            selection[length_tokens > threshold] = "C"
            selection[length_tokens <= threshold] = "N"
            #print(selection)
            predictions.append([instance, selection.tolist()])
        tot_tokens = 0

        for j, labels in enumerate(dev_labels):
            tokens = labels.split(" ")
            instance_true = [tokens[i] for i in range(len(tokens))]

            if "N\n" in instance_true:
                index = instance_true.index("N\n")

            if "C\n" in instance_true:
                index = instance_true.index("C\n")

            instance_true[index] = instance_true[index].replace("\n", "")
            tot_tokens += len(tokens)

            accuracy += sum(np.array(predictions[j][1]) == np.array(instance_true))

        thresholds_tried.append(threshold)
        thresholds_results.append(accuracy/tot_tokens)

    print(np.round(thresholds_results, 3))
    print("best threshold, accuracy on dev set: ", thresholds_tried[np.argmax(np.array(thresholds_results))], np.round(max(thresholds_results), 3))
    return thresholds_tried[np.argmax(np.array(thresholds_results))]

def length_baseline(threshold, testinput, testlabels):

    predictions = []
    accuracy = 0

    for instance in testinput:
        tokens = instance.split(" ")
        length_tokens = np.array([len(t) for t in tokens])

        selection = np.zeros(len(length_tokens), dtype="object")

        selection[length_tokens > threshold] = "C"
        selection[length_tokens <= threshold] = "N"
        # print(selection)
        predictions.append([instance, selection.tolist()])

    tot_tokens = 0

    for j, labels in enumerate(testlabels):
        tokens = labels.split(" ")
        instance_true = [tokens[i] for i in range(len(tokens))]

        if "N\n" in instance_true:
            index = instance_true.index("N\n")

        if "C\n" in instance_true:
            index = instance_true.index("C\n")

        instance_true[index] = instance_true[index].replace("\n", "")
        tot_tokens += len(tokens)
        accuracy += sum(np.array(predictions[j][1]) == np.array(instance_true))

    return accuracy/tot_tokens , predictions

def test_frequency_baseline(dev_sentences, dev_labels):
    thresholds_tried = []
    thresholds_results = []

    for i in np.linspace(0, 0.0005, 201).tolist():
        threshold = i
        predictions = []
        accuracy = 0

        for instance in dev_sentences:
            tokens = instance.split(" ")
            frequency_tokens = np.array([word_frequency(t, "en") for t in tokens])

            selection = np.zeros(len(frequency_tokens), dtype="object")

            selection[frequency_tokens > threshold] = "N"
            selection[frequency_tokens <= threshold] = "C"

            predictions.append([instance, selection.tolist()])
        tot_tokens = 0

        for j, labels in enumerate(dev_labels):
            tokens = labels.split(" ")
            instance_true = [tokens[i] for i in range(len(tokens))]

            if "N\n" in instance_true:
                index = instance_true.index("N\n")

            if "C\n" in instance_true:
                index = instance_true.index("C\n")

            instance_true[index] = instance_true[index].replace("\n", "")
            tot_tokens += len(tokens)
            accuracy += sum(np.array(predictions[j][1]) == np.array(instance_true))

        thresholds_tried.append(threshold)
        thresholds_results.append(accuracy / tot_tokens)

    print(np.round(thresholds_results, 4))

    print("best threshold, accuracy on dev set: ", thresholds_tried[np.argmax(np.array(thresholds_results))],
          np.round(max(thresholds_results), 4))

    return thresholds_tried[np.argmax(np.array(thresholds_results))]

def frequency_baseline(threshold, testinput, testlabels):
    predictions = []
    accuracy = 0

    for instance in testinput:
        tokens = instance.split(" ")
        frequency_tokens = np.array([word_frequency(t, "en") for t in tokens])

        selection = np.zeros(len(frequency_tokens), dtype="object")

        selection[frequency_tokens > threshold] = "N"
        selection[frequency_tokens <= threshold] = "C"
        # print(selection)
        predictions.append([instance, selection.tolist()])
    tot_tokens = 0

    for j, labels in enumerate(testlabels):
        tokens = labels.split(" ")
        instance_true = [tokens[i] for i in range(len(tokens))]

        if "N\n" in instance_true:
            index = instance_true.index("N\n")

        if "C\n" in instance_true:
            index = instance_true.index("C\n")

        instance_true[index] = instance_true[index].replace("\n", "")
        tot_tokens += len(tokens)
        accuracy += sum(np.array(predictions[j][1]) == np.array(instance_true))

    return accuracy/tot_tokens, predictions


def write_to_tsv(output_file, true_test_labels, prediction_data):
    with open(output_file, "w") as f:
        for i in range(len(prediction_data)):
            words = prediction_data[i][0].split()

            for j, word in enumerate(words):
                prediction = prediction_data[i][1][j]
                if word != ".":
                    f.write("\t".join([word, true_test_labels[i][j], prediction]))
                    f.write("\n")
                else:
                    f.write("\t".join(['.', true_test_labels[i][j], prediction]))
                    f.write("\n")
                    f.write("----------\n")
    return 0


if __name__ == '__main__':
    np.random.seed(10)
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "sentences.txt", encoding="UTF8") as sent_file:
        train_sentences = sent_file.readlines()


    with open(train_path + "labels.txt", encoding="UTF8") as label_file:
        train_labels = label_file.readlines()


    with open(dev_path + "sentences.txt", encoding="UTF8") as dev_file:
        dev_sentences = dev_file.readlines()

    with open(dev_path + "labels.txt") as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "sentences.txt") as testfile:
        testinput = testfile.readlines()

    with open(test_path + "labels.txt", encoding="UTF8") as test_labelfile:
        testlabels = test_labelfile.readlines()

    part_B()

    print("-----------------------------------")
    print()
    print("MAJORITY BASELINE:")
    print()
    print("dev data")
    majority_accuracy, majority_predictions = majority_baseline(train_sentences, train_labels, dev_sentences, dev_labels)
    print(majority_accuracy)
    print(majority_predictions)
    print()
    print("test data")
    majority_accuracy, majority_predictions = majority_baseline(train_sentences, train_labels, testinput, testlabels)
    print(majority_accuracy)
    print(majority_predictions)
    print()
    print("-----------------------------------")
    print()
    print("RANDOM BASELINE: ")
    print()
    print("dev data")
    random_accuracy, random_predictions = random_baseline(train_sentences, train_labels, dev_sentences, dev_labels)
    print(random_accuracy)
    print(random_predictions)
    print()
    print("test data")
    random_accuracy, random_predictions = random_baseline(train_sentences, train_labels, testinput, testlabels)
    print(random_accuracy)
    print(random_predictions)
    print()
    print("-----------------------------------")
    print()
    print("LENGTH BASELINE: ")
    print()
    print("dev data")
    best_length_threshold =test_length_baseline(dev_sentences, dev_labels)
    print()
    print("test data")
    length_accuracy, length_predictions =length_baseline(best_length_threshold, testinput, testlabels)
    print(length_accuracy)
    print(length_predictions)
    print()
    print("-----------------------------------")
    print("FREQUENCY BASELINE: ")
    print()
    print("dev data")
    best_frequency_threshold = test_frequency_baseline(dev_sentences, dev_labels)
    print()
    print("test data")
    frequency_accuracy, frequency_predictions = frequency_baseline(best_frequency_threshold, testinput, testlabels)
    print(frequency_accuracy)
    print(frequency_predictions)

    # TODO: output the predictions in a suitable way so that you can evaluate them
    true_labels = []
    for line in testlabels:
        true_labels.append(line[:-1].split())

    print(true_labels)
    outfile = "./experiments/base_model/frequency_output.tsv"
    test_metrics = write_to_tsv(outfile, true_labels, frequency_predictions)

    outfile = "./experiments/base_model/majority_output.tsv"
    test_metrics = write_to_tsv(outfile, true_labels, majority_predictions)

    outfile = "./experiments/base_model/random_output.tsv"
    test_metrics = write_to_tsv(outfile, true_labels, random_predictions)

    outfile = "./experiments/base_model/length_output.tsv"
    test_metrics = write_to_tsv(outfile, true_labels, length_predictions)


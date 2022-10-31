# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt

import spacy
from spacy import displacy
import os
import numpy as np
import textacy
import pandas as pd
from wordfreq import word_frequency
import matplotlib.pyplot as plt
from collections import Counter


def n_grams_func(n_size, ls):
    n_gram = list()

    for index in range(len(ls) - n_size):
        n_gram.append(" ".join(ls[index: index + n_size]))

    return n_gram


def part_A(doc):

    number_of_tokens = 0
    types = set()
    number_of_sentences = 0
    word_frequencies = Counter()
    number_of_tokens2 = 0
    characters = 0

    for sentence in doc.sents:
        number_of_sentences += 1
        words = []
        for token in sentence:
            types.add(token.text)
            number_of_tokens2 += 1
            # Let's filter out punctuation
            if token.is_alpha:
                words.append(token.text)
                characters += len(token.text)
        word_frequencies.update(words)

    num_words = sum(word_frequencies.values())
    print("PART A: 1")
    print("#tokens", number_of_tokens2)
    print("#Types", len(types))
    print("#Num_words", num_words)
    print("#number_of_sentences", number_of_sentences)
    print("avg words per sentence", num_words / number_of_sentences)
    print("avg characters per word", characters / num_words)

    word_classes = {}
    for token in doc:
        if token.text != '\n':
            if token.tag_ not in word_classes:
                tag_frequency = 0
                Relative_Tag = 0
                types = {}
                word_classes[token.tag_] = [token.pos_, token.tag_, tag_frequency, types, Relative_Tag]

            word_classes[token.tag_][2] += 1
            types = word_classes[token.tag_][3]

            if token.text in types:
                types[token.text] += 1
            else:
                types[token.text] = 1

            word_classes[token.tag_][3] = types

    total_tags = 0

    for key in word_classes:
        word_classes[key][3] = {k: v for k, v in
                                sorted(word_classes[key][3].items(), key=lambda item: item[1], reverse=True)}
        word_classes[key].append(list(word_classes[key][3])[-1])
        word_classes[key][3] = list(word_classes[key][3])[0:3]

        total_tags += word_classes[key][2]

    for key in word_classes:
        word_classes[key][4] = word_classes[key][2] / total_tags

    word_classes = {k: v for k, v in sorted(word_classes.items(), key=lambda item: item[1][2], reverse=True)}

    print("PART A: 2")
    print(list(word_classes.values())[0:10])

    bi_grams = list(textacy.extract.basics.ngrams(doc, 2, min_freq=2))
    tri_grams = list(textacy.extract.basics.ngrams(doc, 3, min_freq=2))
    bi_grams_dict = dict()
    tri_grams_dict = dict()
    for element in bi_grams:
        if element.text in bi_grams_dict:
            bi_grams_dict[element.text] += 1
        else:
            bi_grams_dict[element.text] = 1

    print("PART A: 3")
    for element in tri_grams:
        if element.text in tri_grams_dict:
            tri_grams_dict[element.text] += 1
        else:
            tri_grams_dict[element.text] = 1

    bi_grams_dict = {k: v for k, v in sorted(bi_grams_dict.items(), key=lambda item: item[1], reverse=True)}
    bi_grams_3 = list(bi_grams_dict)[:3]
    print("Token bi_grams:", bi_grams_3)

    tri_grams_dict = {k: v for k, v in sorted(tri_grams_dict.items(), key=lambda item: item[1], reverse=True)}
    tri_grams_3 = list(tri_grams_dict)[:3]
    print("Token tri_grams:", tri_grams_3)

    pos_list = list()

    for token in doc:
        pos_list.append(token.pos_)

    bi_grams = n_grams_func(2, pos_list)
    tri_grams = n_grams_func(3, pos_list)

    bi_grams_dict = dict()
    tri_grams_dict = dict()

    for element in bi_grams:
        if element in bi_grams_dict:
            bi_grams_dict[element] += 1
        else:
            bi_grams_dict[element] = 1

    for element in tri_grams:
        if element in tri_grams_dict:
            tri_grams_dict[element] += 1
        else:
            tri_grams_dict[element] = 1

    bi_grams_dict = {k: v for k, v in sorted(bi_grams_dict.items(), key=lambda item: item[1], reverse=True)}
    bi_grams_3 = list(bi_grams_dict)[:3]
    print("POS bi_grams:", bi_grams_3)

    tri_grams_dict = {k: v for k, v in sorted(tri_grams_dict.items(), key=lambda item: item[1], reverse=True)}
    tri_grams_3 = list(tri_grams_dict)[:3]
    print("POS bi_grams:", tri_grams_3)



def ent_freq(doc):

    name_label = []
    name_ent = []
    for sentence in doc.sents:

        for ent in sentence.ents:
            name_ent.append(ent.text)
            name_label.append(ent.label_)

    num_ent = len(name_ent)
    num_ent_unique=len(np.unique(np.array(name_ent)))
    unique=np.unique(np.array(name_label))


    print("Number of entities/Number of unique entities/Number of different entity labels",num_ent,num_ent_unique, len(unique))



if __name__ == "__main__":
    f = open('data/preprocessed/train/sentences.txt', 'r', encoding="utf8").read()
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(f)

    part_A(doc)
    part_B()

    ent_freq(doc)

    text = "children are thought to be aged three , eight , and ten years , alongside an eighteen-month-old baby .\
    We mixed different concentrations of ROS with the spores , plated them out on petridishes with an agar-solution where fungus can grow on . \
    They feel they are under-represented in higher education and are suffering in a regional economic downturn .\
     Especially as it concerns a third party building up its military presence near our borders .\
      Police said three children were hospitalised for \" severe dehydration \" ."
    doc_ent = nlp(text)
    displacy.serve(doc_ent, style='dep')
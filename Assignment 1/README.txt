This project contains code for part 1 of the course Natural Language Processing Technology 2022 at VU Amsterdam put together by Lisa Beinborn.

References:
The modelling part draws substantially on the code project for the Stanford course cs230-stanford developed by Surag Nair, Guillaume Genthial and Olivier Moindrot (https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp).
It has been simplified and adapted by Lisa Beinborn.

The data in data/original is a small subset from http://sites.google.com/view/cwisharedtask2018/. Check data/original/README.md for details.

The data in data/preprocessed has been processed by Sian Gooding for her submission to the shared task (https://github.com/siangooding/cwi/tree/master/CWI%20Sequence%20Labeller).


Your task:
Make sure to install the libraries in requirements.txt.

- For part A of the assignment, you need to provide linguistic analyses in TODO_analyses.py using spacy.
- For part B of the assignment, you need to calculate baselines in TODO_baselines.py. The existing code snippet is supposed to provide some guidance but you can delete it and start from scratch if you find it irritating. Keep in mind to determine the thresholds for the baselines on the training data.
- For part C of the assignment, you need to build the vocabulary and train the model. Inspect the code to understand what is happening. Implement functions to evaluate the output of the model and the baselines in TODO_detailed_evaluation.py.

Your code will not be graded but it should be well documented and support the results of your submission. Your submission is only complete with the code. We will sporadically check the completeness and quality of the code.


Further information:

Names of the authors: 
- Michele Belloli
- Abel van Gennep
- Nasrin Rastgoo

Content of the folder:

The folder contains all the files given from the initial download zip file.
Furthermore, we have several .tsv files that contain the recorded data from several fitted models.
- we have files related to the evaluation of hyperparameter values named as: {num_epochs}epochs.tsv
- we have length/frequency/majority/random_output.tsv which are the results of our baseline models
These files will be used in the TODO_detailed_evaluation.py file to return the details part C of the assignment.
The file TODO_analyses.py contains part A of the assignment
The file TODO_baselines.py contains part B of the assignment
import numpy as numpy
import matplotlib.pyplot as py
import string
from sklearn.model_selection import train_test_split

input_files = [
    'Markov-Models/gene_wolfe_sample.txt',
    'Markov-Models/john_steinbeck_sample.txt'
]

input_texts = []
labels = []

for label, f in enumerate(input_files):
    print(f"{f} corresponds to label {label}")

    with open(f, 'r') as file:
        for line in file:
            line = line.rstrip().lower()
            if line:
                line = line.translate(str.maketrans('', '', string.punctuation))
                input_texts.append(line)
                labels.append(label)

train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels)

print(len(Ytrain)), print(len(Ytest))
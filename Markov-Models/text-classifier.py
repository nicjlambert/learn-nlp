import numpy as np
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

idx = 1
word2idx = {'<unk>': 0}

for text in train_text:
    tokens = text.split()
    for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1

#print(word2idx)

train_text_int = []
test_text_int = []

for text in train_text:
    tokens = text.split()
    line_as_int = [word2idx[token] for token in tokens]
    train_text_int.append(line_as_int)

for text in test_text:
    tokens = text.split()
    line_as_int = [word2idx.get(token, 0) for token in tokens]
    test_text_int.append(line_as_int)

# initialise A and pi matrices - for both classes
V = len(word2idx)

A0 = np.ones((V, V))
pi0 = np.ones(V)

A1 = np.ones((V, V))
pi1 = np.ones(V)

# compute counts for A and pi
def compute_counts(text_as_int, A, pi):
    for tokens in text_as_int:
        last_idx = None
        for idx in tokens:
            if last_idx is None:
                pi[idx] += 1
            else:
                A[last_idx, idx] += 1
            
            #update the last idx
            last_idx = idx

compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0], A0, pi0)
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 1], A1, pi1)

A0 /= A0.sum(axis=1, keepdims=True)
pi0 /= pi0.sum()

A1 /= A1.sum(axis=1, keepdims=True)
pi1 /= pi1.sum()

# log A and pi since we don't need the actual probabilities
logA0 = np.log(A0)
logpi0 = np.log(pi0)

logA1 = np.log(A1)
logpi1 = np.log(pi1)

# compute priors
count0 = sum(y == 0 for y in Ytrain)
count1 = sum(y == 1 for y in Ytrain)
total = len(Ytrain)

p0 = count0 / total
p1 = count1 / total

logp0 = np.log(p0)
logp1 = np.log(p1)

print(p0, p1)

# build a classifier
class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K = len(logpriors)
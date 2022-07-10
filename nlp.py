import pandas as pd
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

# resolve error install <wordnet, omw-1.4> with following:
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download("stopwords")

with open('MBS.txt') as f:
    lines = f.readlines()
#lemmatizer = WordNetLemmatizer()

#words = ['articles', 'friendship', 'studies', 'phones']
#for word in words:
   # print(lemmatizer.lemmatize(word))

stop_words = set(stopwords.words("english"))

# create an empty list to hold the words that make it past the filter
filtered_list = []

for word in lines:
    if word not in stop_words:
        filtered_list.append(word)
print(stop_words)
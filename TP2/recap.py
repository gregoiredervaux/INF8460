import os
import re
import nltk
import collections
import numpy as np
from nltk.corpus import stopwords
nltk.download('stopwords')

##################### Question 1 #####################
## lecture

global dictionnaire

dictionnaire = {
    "test": {"pos": {},
            "neg": {}},
    "train": {"pos": {},
            "neg": {}}
}
nb_failed_files = 0
for train_type in ["test", "train"]:
    for classification in ["pos", "neg"]:
        path = './data/' + train_type + '/' + classification + '/'
        for file in os.listdir(path):
            id_review, rate_review = file.split("_")
            rate_review = rate_review.split(".txt")[0]
            if int(id_review) not in dictionnaire[train_type][classification]:
                with open(path + file, "r") as f:
                    try:
                        dictionnaire[train_type][classification][int(id_review)] = { "rate": int(rate_review),
                                                                            "review": f.read()}
                    except:
                        nb_failed_files += 1

print("Nombre de fichiers non-ouverts : " + str(nb_failed_files))

## petit a)

def clean_doc(dictio):

    for type_dataset in dictio:
        for sentiment_type in dictio[type_dataset]:
            for id_review in dictio[type_dataset][sentiment_type]:
                review = dictio[type_dataset][sentiment_type][id_review]["review"]
                review = review.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,./?\|`~-=_+"})
                review = review.split()
                stop_words = set(stopwords.words('english'))
                review_list = [w for w in review if not w in stop_words]
                segmentize_review = []

                for word in review_list:
                    if(re.match('^[a-zA-Z]+$', word) and len(word)>1):
                        segmentize_review.append(word)

                dictio[type_dataset][sentiment_type][id_review]["review"] = segmentize_review

clean_doc(dictionnaire)
print(dictionnaire["test"]["pos"][0]["review"])


##b)
def build_voc(dico_train):
    count = collections.defaultdict(lambda: 0)
    for classi in dico_train:
        for review in dico_train[classi]:
            text = dico_train[classi][review]["review"]
            for word in text:
                if word in count:
                    count[word] += 1
                else:
                    count[word] = 1
    n = 0
    f = open("./data/vocab.txt", "w+")
    for word in count:
        if count[word] >= 5:
            n += 1
            f.write(word + "\n")
    print("Nombre de mots dans le vocabulaire : " + str(n))
    f.close()
    return n

build_voc(dictionnaire["train"])


## c)
def get_top_unigrams(n):

    dico_count = {}

    for train_type in ["test", "train"]:
        for classification in ["pos", "neg"]:
            for review in dictionnaire[train_type][classification]:
                for word in dictionnaire[train_type][classification][review]['review']:
                    if word not in dico_count:
                        dico_count[word] = 1
                    else:
                        dico_count[word] += 1

    counter = collections.Counter(dico_count)
    top_unigrams = counter.most_common(n)
    return [couple[0] for couple in top_unigrams]

print(get_top_unigrams(10))

## d)

def get_top_unigrams_per_cls(n, cls):
    dico_count = {}

    for train_type in ["test", "train"]:
        for review in dictionnaire[train_type][cls]:
            for word in dictionnaire[train_type][cls][review]['review']:
                if word not in dico_count:
                    dico_count[word] = 1
                else:
                    dico_count[word] += 1

    counter = collections.Counter(dico_count)
    top_unigrams = counter.most_common(n)
    return [couple[0] for couple in top_unigrams]


print(get_top_unigrams_per_cls(10, "pos"))
print(get_top_unigrams_per_cls(10, "neg"))

##################### Question 2 #####################

unigrams = get_top_unigrams(5000)
print(unigrams[0:30])
BoW = []
for train_type in ["test", "train"]:
    for classification in ["pos", "neg"]:
        for review in dictionnaire[train_type][classification]:
            new_ligne = np.zeros(len(unigrams))
            for word in dictionnaire[train_type][classification][review]['review']:
                if word in unigrams:
                    new_ligne[unigrams.index(word)] = 1
            BoW.append(new_ligne)
## Pour la dernière question, peut être essayer une réduction de dimension

import os
import re
import nltk
import collections
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
                review = review.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
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

clean_doc(dictionnaire)
build_voc(dictionnaire["train"])
# print(dictionnaire_segm["test"]["pos"][0]["review"])


## c)
def get_top_unigrams(n):
    unigrams = []
    count = []

    for train_type in ["test", "train"]:
        for classification in ["pos", "neg"]:
            for review in dictionnaire[train_type][classification]:
                for word in dictionnaire[train_type][classification][review]['review']:
                    if word not in unigrams:
                        unigrams.append(word)
                        count.append(1)
                    else:
                        count[unigrams.index(word)] += 1

    top_unigram = []
    for i in range(n):
        max_index = count.index(max(count))
        top_unigram.append(unigrams[max_index])
        del unigrams[max_index]
        del count[max_index]

    return top_unigram

print(get_top_unigrams(10))

## d)

def get_top_unigrams_per_cls(n, cls):
    unigrams = []
    count = []
    for train_class in ["train", "test"]:
        for review in dictionnaire[train_class][cls]:
            for word in dictionnaire[train_class][cls][review]['review']:
                if word not in unigrams:
                    unigrams.append(word)
                    count.append(1)
                else:
                    count[unigrams.index(word)] += 1

    top_unigram = []
    for i in range(n):
        max_index = count.index(max(count))
        top_unigram.append(unigrams[max_index])
        del unigrams[max_index]
        del count[max_index]

    return top_unigram


print(get_top_unigrams_per_cls(10, "pos"))
print(get_top_unigrams_per_cls(10, "neg"))

##################### Question 2 #####################


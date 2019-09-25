import os
import re
from nltk.corpus import stopwords
import string

##################### Question 1 #####################
## lecture

global dictionnaire
dictionnaire = {
    "test": {"pos": {},
            "neg": {}},
    "train": {"pos": {},
            "neg": {}}
}
for train_type in ["test", "train"]:
    for classification in ["pos", "neg"]:
        path = './data/' + train_type + '/' + classification + '/'
        for file in os.listdir(path):
            id_review, rate_review = file.split("_")
            rate_review = rate_review.split(".txt")[0]
            if int(id_review) not in dictionnaire[train_type][classification]:
                with open(path + file, "r") as f:
                    dictionnaire[train_type][classification][int(id_review)] = { "rate": int(rate_review),
                                                                            "review": f.read()}

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


dictionnaire_segm = dictionnaire.copy()
clean_doc(dictionnaire_segm)
print(dictionnaire_segm["test"]["pos"][0]["review"])

# test dictionnaire

"""
dictionnaire = {
    "test":
        {"pos":
            { 0:
                    { "rate": 2,
                      "review": "lol"},
              1:
                    { "rate": 2,
                      "review": "lol"},
              3:
                    { "rate": 2,
                      "review": "lol"},
              4:
                    { "rate": 2,
                      "review": "lol"},
              5:
                    { "rate": 2,
                      "review": "lol"}

            },
        "neg":
        { 0:
                    { "rate": 2,
                      "review": "zoupla"},
              1:
                    { "rate": 2,
                      "review": "quick"},
              3:
                    { "rate": 2,
                      "review": "toupitou"},
              4:
                    { "rate": 2,
                      "review": "zoro"},
              5:
                    { "rate": 2,
                      "review": "oreille"}

            }},
    "train": {"pos": {},
            "neg": {}}
}
"""

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

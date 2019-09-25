import os
import re
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


##################### Question 2 #####################

from nltk.corpus import stopwords
import string

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

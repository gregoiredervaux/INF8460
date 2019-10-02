import os
import re
import nltk
import collections
import math
import itertools
import time
import numpy as np
from scipy.spatial import distance
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
print("les 5000 premiers unigrams")
print(unigrams[0:30])
debut_mm = time.time()
BoW = np.array([])
for train_type in ["test", "train"]:
    for classification in ["pos", "neg"]:
        # on construit un petit BoW, pur le type et la classe des documents
        # pour eviter les appends a chaque nouvelle ligne et profiter des types numpy
        BoW_part = np.zeros((len(dictionnaire[train_type][classification]), len(unigrams)))
        for index, review in enumerate(dictionnaire[train_type][classification]):
            for word in dictionnaire[train_type][classification][review]['review']:
                if word in unigrams:
                    BoW_part[index, unigrams.index(word)] += 1

        BoW = np.append(BoW, BoW_part, axis=0) if BoW.size != 0 else BoW_part

print("\nBag of Word")
print("fait en : " + str(time.time() - debut_mm) + "s")
print(BoW)

debut_mm = time.time()
TFIDF_BoW = BoW.copy()
# liste des idf par index (cf unigrams)
idf = []
for i,_ in enumerate(unigrams):
    dfi = 0
    for j in range(TFIDF_BoW.shape[0]):
        # on compte le nombre de documents qui contiennent ce mot
        dfi += 1 if TFIDF_BoW[j, i] != 0 else 0
    # on calcul et ajoute le idf a la liste des idf
    idf.append(math.log(TFIDF_BoW.shape[0] / dfi))

# on modifie la matrice
for j in range(TFIDF_BoW.shape[0]):
    for i, _ in enumerate(unigrams):
        value = TFIDF_BoW[j, i] * idf[i]
        TFIDF_BoW[j, i] = value
print("\nBag of Word TFIDF")
print("fait en : " + str(time.time() - debut_mm) + "s")
print(TFIDF_BoW)


debut_mm = time.time()
matrice_MM = np.zeros((len(unigrams), len(unigrams)))
for train_type in ["test", "train"]:
    for classification in ["pos", "neg"]:
        for review in dictionnaire[train_type][classification]:
            # on selectionne seulement les mots qui font parti des 5000 selectionnés
            words_to_consider  = list(set(unigrams).intersection(dictionnaire[train_type][classification][review]['review']))
            index_to_consider = [unigrams.index(couple_word) for couple_word in words_to_consider]
            permutation = itertools.permutations(index_to_consider, 2)
            for index_to_consider in permutation:
                matrice_MM[index_to_consider] += 1

print("matrice_MM")
print("fait en : " + str(time.time() - debut_mm) + "s")
print(matrice_MM)

debut_mm = time.time()
def calculate_PPMI(matrice_MM):
    colsum = np.sum(matrice_MM, axis=0)
    row_sum = np.sum(matrice_MM, axis=1)
    sum = np.sum(matrice_MM)
    expected = np.zeros((matrice_MM.shape[0], matrice_MM.shape[1]))
    ppmi =  np.zeros((matrice_MM.shape[0], matrice_MM.shape[1]))
    for index,_ in np.ndenumerate(expected):
        expected[index] = (row_sum[index[0]] * colsum[index[1]]) / sum
        pmi = math.log(matrice_MM[index] / expected[index]) if matrice_MM[index] > 0 else 0
        ppmi[index] = pmi if pmi > 0 else 0

    return ppmi

print("matrice_PPMI")
matrice_PPMI = calculate_PPMI(matrice_MM)
print("fait en : " + str(time.time() - debut_mm) + "s")
print(matrice_PPMI)

##################### Question 3 #####################

def get_euclidean_distance(v1 ,v2):
    return distance.euclidean(v1, v2)

def get_cosinus_distance(v1, v2):
    return distance.cosine(v1, v2)

def get_most_similar_PPMI(word, metric, n):
    index_word = unigrams.index(word)
    ligne_word = matrice_PPMI[index_word]
    vector_distance = {}
    i = 0
    if metric == "cosinus":
        for index, ligne in enumerate(matrice_PPMI):
            vector_distance[unigrams[index]] = get_cosinus_distance(ligne_word, ligne)
            i += 1
    elif metric == "euclidean":
        for index, ligne in enumerate(matrice_PPMI):
            vector_distance[unigrams[index]] = get_euclidean_distance(ligne_word, ligne)
            i += 1
    else:
        print("metric inconnue")

    counter = collections.Counter(vector_distance)
    top_unigrams = counter.most_common(n)
    return [couple[0] for couple in top_unigrams]

debut_mm = time.time()
print("\nles mots les plus proches de \"bad\": ")
most_similar = get_most_similar_PPMI("bad", "euclidean", 5)
print(most_similar)
print("fait en : " + str(time.time() - debut_mm) + "s")


def get_most_similar_TFIDF(word, metric, n):
    index_word = unigrams.index(word)
    column_word = TFIDF_BoW[:, index_word]
    vector_distance = {}
    if metric == "cosinus":
        for index, column in enumerate(TFIDF_BoW.T):
            vector_distance[unigrams[index]] = get_cosinus_distance(column_word, column)
    elif metric == "euclidean":
        for index, column in enumerate(TFIDF_BoW.T):
            vector_distance[unigrams[index]] = get_euclidean_distance(column_word, column)
    else:
        print("metric inconnue")

    counter = collections.Counter(vector_distance)
    top_unigrams = counter.most_common(n)
    return [couple[0] for couple in top_unigrams]

debut_mm = time.time()
print("les mots les plus proches de \"bad\": ")
most_similar = get_most_similar_TFIDF("bad", "euclidean", 5)
print(most_similar)
print("fait en : " + str(time.time() - debut_mm) + "s")




## Pour la dernière question, peut être essayer une réduction de dimension

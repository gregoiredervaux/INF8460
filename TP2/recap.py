import os
import re
import nltk
import collections
import math
import itertools
import time
import copy
import numpy as np
import matplotlib as plt
from scipy.spatial import distance
from nltk.corpus import stopwords
from collections import *
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
        path = './fakeData/' + train_type + '/' + classification + '/'
        for file in os.listdir(path):
            id_review, rate_review = file.split("_")
            rate_review = rate_review.split(".txt")[0]
            if int(id_review) not in dictionnaire[train_type][classification]:
                with open(path + file, "r") as f:
                    try:
                        dictionnaire[train_type][classification][int(id_review)] = { "rate": int(rate_review),
                                                                            "review_str": f.read()}
                    except:
                        nb_failed_files += 1

print("Nombre de fichiers non-ouverts : " + str(nb_failed_files))
init_dictionnaire = copy.deepcopy(dictionnaire)


## petit a)

def clean_doc(dictio):

    for type_dataset in dictio:
        for sentiment_type in dictio[type_dataset]:
            for id_review in dictio[type_dataset][sentiment_type]:
                review = dictio[type_dataset][sentiment_type][id_review]["review_str"]
                review = review.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,./?\|`~-=_+"})
                review = review.lower()
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

taille_voc = build_voc(dictionnaire["train"])

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
            vector_distance[unigrams[index]] = -get_cosinus_distance(ligne_word, ligne)
            i += 1
    elif metric == "euclidean":
        for index, ligne in enumerate(matrice_PPMI):
            vector_distance[unigrams[index]] = -get_euclidean_distance(ligne_word, ligne)
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
            vector_distance[unigrams[index]] = -get_cosinus_distance(column_word, column)
    elif metric == "euclidean":
        for index, column in enumerate(TFIDF_BoW.T):
            vector_distance[unigrams[index]] = -get_euclidean_distance(column_word, column)
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

##################### Question 4 ###################

# def train_char_lm(data, order=4):
#     lm = defaultdict(Counter)
#     voc = set()
#     for review_index in data:
#         review = data[review_index]["review_str"]
#         # comme dans l'article on marque le début d'un document
#         pad = "~" * order
#         padded_review = pad + review
#         for i in range(len(padded_review) - order):
#             history, char = padded_review[i:i + order], padded_review[i + order]
#             if char not in voc: voc.add(char)
#             lm[history][char] += 1
#
#     def normalize(counter):
#         s = float(sum(counter.values()))
#         unsmooth = {}
#         for c, cnt in counter.items():
#             unsmooth[c] = math.log((cnt + 1) / (s + len(voc)))
#         unsmooth["not_found"] = math.log(1 / (s + len(voc)))
#         return unsmooth
#
#     outlm = {hist: normalize(chars) for hist, chars in lm.items()}
#     outlm["not_found"] = math.log(1 / len(voc))
#     return outlm
#
#
# def calcul_proba_char(document, lm, order=3):
#     proba = 0
#     for i in range(len(document) - order):
#         history, word = document[i:i + order], document[i + order]
#         if history in lm:
#             if word in lm[history]:
#                 proba += lm[history][word]
#             else:
#                 proba += lm[history]["not_found"]
#         else:
#             proba += lm["not_found"]
#     return proba
# # on crée un dictionnaire pour stocker nos prédictions
# pred = {
#     "clean" : { "pos": {},
#               "neg": {}},
#     "raw": { "pos": {},
#               "neg": {}}
#        }
#
# pos = []
# neg = []
# print("sur le dictionnaire propre")
# n = 10
# for i in range(3, n):
#     lm_pos = train_char_lm(dictionnaire["train"]["pos"], i)
#     lm_neg = train_char_lm(dictionnaire["train"]["neg"], i)
#     for classe in init_dictionnaire["train"]:
#         pred["clean"][classe][i] = []
#         class_by_index = []
#         nb_pos = 0
#         for index_doc in range(len(dictionnaire["test"]["pos"])):
#             proba_pos = calcul_proba_char(dictionnaire["test"][classe][index_doc]["review_str"], lm_pos, i)
#             proba_neg = calcul_proba_char(dictionnaire["test"][classe][index_doc]["review_str"], lm_neg, i)
#             if proba_pos > proba_neg:
#                 nb_pos += 1
#                 pred["clean"][classe][i].append(1)
#             else:
#                 pred["clean"][classe][i].append(0)
#
#         if classe == "pos":
#             pos.append(nb_pos / len(dictionnaire["test"]["pos"]))
#         else:
#             neg.append((len(dictionnaire["test"]["pos"]) - nb_pos) / len(dictionnaire["test"]["pos"]))
#
# plt.plot(list(range(3, n)), pos, label="class positif")
# plt.plot(list(range(3,n)), neg, label="class négatif")
# plt.xlabel('orders')
# plt.ylabel('précision')
# plt.legend()
# plt.show()
#
#
# pos = []
# neg = []
# print("\nsur le dictionnaire sans modification:")
# for i in range(3, n):
#     class_by_index = []
#     nb_pos = 0
#     lm_pos = train_char_lm(init_dictionnaire["train"]["pos"], i)
#     lm_neg = train_char_lm(init_dictionnaire["train"]["neg"], i)
#     for classe in init_dictionnaire["train"]:
#         pred["raw"][classe][i] = []
#         class_by_index = []
#         nb_pos = 0
#         for index_doc in range(len(dictionnaire["test"][classe])):
#             proba_pos = calcul_proba_char(init_dictionnaire["test"][classe][index_doc]["review_str"], lm_pos, i)
#             proba_neg = calcul_proba_char(init_dictionnaire["test"][classe][index_doc]["review_str"], lm_neg, i)
#             if proba_pos > proba_neg:
#                 nb_pos += 1
#                 pred["raw"][classe][i].append(1)
#             else:
#                 pred["raw"][classe][i].append(0)
#
#         if classe == "pos":
#             pos.append(nb_pos / len(dictionnaire["test"]["pos"]))
#         else:
#             neg.append((len(dictionnaire["test"]["pos"]) - nb_pos) / len(dictionnaire["test"]["pos"]))
#
# plt.plot(list(range(3,n)), pos, label="class positif")
# plt.plot(list(range(3,n)),neg, label="class négatif")
# plt.xlabel('orders')
# plt.ylabel('précision')
# plt.legend()
# plt.show()


################## suite Question 1 #################




##################### Question 5 #####################
from sklearn.naive_bayes import MultinomialNB


# y = 0:pos 1:neg
X_test = TFIDF_BoW[0:len(dictionnaire["test"]["pos"])+len(dictionnaire["test"]["neg"])][:]
y_test_nb = np.full((1, len(dictionnaire["test"]["pos"])), 0)
y_test_nb = np.append(y_test_nb, np.full((1, len(dictionnaire["test"]["neg"])), 1))

X_train = TFIDF_BoW[len(dictionnaire["test"]["pos"])+len(dictionnaire["test"]["neg"]):][:]
y_train = np.full((1, len(dictionnaire["train"]["pos"])), 0)
y_train = np.append(y_train, np.full((1, len(dictionnaire["train"]["neg"])), 1))

classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred_nb = classifier.predict(X_test)
print(y_pred_nb)

##################### Question 6 #####################
from sklearn.decomposition import TruncatedSVD

SVD = TruncatedSVD(100)
new_X_train = np.matrix(SVD.fit_transform(X_train))
new_X_train = np.subtract(new_X_train, new_X_train.min())
classifier.fit(new_X_train, y_train)
new_X_test = np.matrix(SVD.fit_transform(X_test))
new_X_test = np.subtract(new_X_test, new_X_test.min())
y_pred_svdNB = classifier.predict(new_X_test)

classifier = MultinomialNB()
PPMI_doc = calculate_PPMI(BoW)
PPMI_doc = [[el+1 for el in row] for row in PPMI_doc]
X_test = PPMI_doc[0:len(dictionnaire["test"]["pos"])+len(dictionnaire["test"]["neg"])][:]
X_train = PPMI_doc[len(dictionnaire["test"]["pos"])+len(dictionnaire["test"]["neg"]):][:]
classifier.fit(X_train, y_train)
y_pred_multinomialNB = classifier.predict(X_test)
print(y_pred_multinomialNB)


##################### Question 7 #####################
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def evaluate(y_test, y_pred):
    # accuracy
    print("Accuracy NB:")
    print(accuracy_score(y_test, y_pred))

    # precision
    print("Precision NB:")
    print(precision_score(y_test, y_pred, average='binary'))

    # recall
    print("Recall NB:")
    print(recall_score(y_test, y_pred, average='binary'))

    # F1-score
    print("F1-score NB:")
    print(f1_score(y_test, y_pred, average='binary'))

print("Evaluation of Naive Bayes Multinomial with TF-IDF:")
evaluate(y_test_nb, y_pred_nb)
print("")
print("Evaluation of Naive Bayes Multinomial with TF-IDF and SVD:")
evaluate(y_test_nb, y_pred_svdNB)
print("")
print("Evaluation of Naive Bayes Multinomial with PPMI:")
evaluate(y_test_nb, y_pred_multinomialNB)
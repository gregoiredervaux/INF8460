"""
Question 1.2.1 à 1.2.7 : implémentation d'un modèle n-gramme MLE.

Les premières fonctions permettent de compter les n-grammes d'un corpus puis de calculer les estimées MLE; vous devrez
les compléter dans les questions 1.2.1 à 1.2.4. Elles sont ensuite encapsulées dans une classe `NgramModel` (déjà écrite),
vous pourrez alors entraîner un modèle n-gramme sur un corpus avec la commande :
>>> lm = NgramModel(corpus, n)

Sauf mention contraire, le paramètre `corpus` désigne une liste de phrases tokenizées
"""
from itertools import chain
from itertools import permutations
from random import choices
import collections
from nltk.lm.preprocessing import pad_both_ends
import operator
import collections

import preprocess_corpus as pre


def extract_ngrams_from_sentence(sentence, n):
    """
    Renvoie la liste des n-grammes présents dans la phrase `sentence`.

    >>> extract_ngrams_from_sentence(["Alice", "est", "là"], 2)
    [("<s>", "Alice"), ("Alice", "est"), ("est", "là"), ("là", "</s>")]

    Attention à la gestion du padding (début et fin de phrase).

    :param sentence: list(str), une phrase tokenizée
    :param n: int, l'ordre des n-grammes
    :return: list(tuple(str)), la liste des n-grammes présents dans `sentence`
    """

    sentence = list(pad_both_ends(sentence, n))

    n_gram_list = []
    for i in range(len(sentence) - n + 1):
        n_gram = []
        for j in range(n):
            n_gram.append(sentence[i + j])
        n_gram_list.append(tuple(n_gram))

    return n_gram_list


def extract_ngrams(corpus, n):
    """
    Renvoie la liste des n-grammes présents dans chaque phrase de `corpus`.

    >>> extract_ngrams([["Alice", "est", "là"], ["Bob", "est", "ici"]], 2)
    [
        [("<s>", "Alice"), ("Alice", "est"), ("est", "là"), ("là", "</s>")],
        [("<s>", "Bob"), ("Bob", "est"), ("est", "ici"), ("ici", "</s>")]
    ]

    :param corpus: list(list(str)), un corpus à traiter
    :param n: int, l'ordre des n-grammes
    :return: list(list(tuple(str))), la liste contenant les listes de n-grammes de chaque phrase
    """

    n_gram_corpus = []
    for sentence in corpus:
        n_gram_corpus.append(extract_ngrams_from_sentence(sentence.copy(), n))
    return n_gram_corpus


def count_ngrams(corpus, n):
    """
    Compte les n-grammes présents dans le corpus.

    Attention, le résultat de la fonction doit gérer correctement les n-grammes inconnus. Pour cela, la classe
    `collections.defaultdict` pourra être utile.

    >>> counts = count_ngrams([["Alice", "est", "là"], ["Bob", "est", "ici"]], 2)
    >>> counts[("est",)]["là"] # Bigramme connu
    1
    >>> counts[("est",)]["Alice"] # Bigramme inconnu
    0

    :param corpus: list(list(str)), un corpus à traiter
    :param n: int, l'ordre de n-grammes
    :return: mapping(tuple(str)->mapping(str->int)), l'objet contenant les comptes de chaque n-gramme
    """

    n_gram_corpus = extract_ngrams(corpus, n)
    nested_dictionnary = collections.defaultdict(lambda: 0)

    for sentence in n_gram_corpus:
        for n_gram in sentence:
            if n_gram[0:n - 1] in nested_dictionnary:
                if n_gram[-1] in nested_dictionnary[n_gram[0:n - 1]]:
                    nested_dictionnary[n_gram[0:n - 1]][n_gram[-1]] += 1
                else:
                    nested_dictionnary[n_gram[0:n - 1]][n_gram[-1]] = 1
            else:
                nested_dictionnary[n_gram[0:n - 1]] = collections.defaultdict(lambda: 0)
                nested_dictionnary[n_gram[0:n - 1]][n_gram[-1]] = 1

    return nested_dictionnary

def compute_MLE(counts):
    """
    À partir de l'objet `counts` produit par la fonction `count_ngrams`, transforme les comptes en probabilités.

    >>> mle_counts = compute_MLE(counts)
    >>> mle_counts[("est",)]["là"] # 1/2
    0.5
    >>> mle_counts[("est",)]["Alice"] # 0/2
    0

    :param counts: mapping(tuple(str)->mapping(str->int))
    :return: mapping(tuple(str)->mapping(str->float))
    """
    for n_gram_begining in counts:
        sum_n_gram = 0
        for n_gram_end in counts[n_gram_begining]:
            sum_n_gram += counts[n_gram_begining][n_gram_end]
        if sum_n_gram != 0:
            for n_gram_end in counts[n_gram_begining]:
                counts[n_gram_begining][n_gram_end] = counts[n_gram_begining][n_gram_end]/sum_n_gram
    return counts

class NgramModel(object):
    def __init__(self, corpus, n):
        """
        Initialise un modèle n-gramme MLE à partir d'un corpus.

        :param corpus: list(list(str)), un corpus tokenizé
        :param n: int, l'ordre du modèle
        """
        counts = count_ngrams(corpus, n)
        self.n = n
        self.vocab = list(set(chain(["<s>", "</s>"], *corpus)))
        self.counts = compute_MLE(counts)

    def proba(self, word, context):
        """
        Renvoie P(word | context) selon le modèle.

        :param word: str
        :param context: tuple(str)
        :return: float
        """
        if context not in self.counts or word not in self.counts[context]:
            return 0.0
        return self.counts[context][word]

    def predict_next(self, context):
        """
        Renvoie un mot w tiré au hasard selon la distribution de probabilité P( w | context)

        :param context: tuple(str), un (n-1)-gramme de contexte
        :return: str
        """
        if self.counts[context] != 0:
            p = [self.counts[context][word] for word in self.counts[context]]
            chosen = choices(list(self.counts[context].keys()), p)
        else:
            chosen=["<Not found in the text>"]
        return chosen[0]



if __name__ == "__main__":
    """
    Pour n=1, 2, 3:
    - entraînez un modèle n-gramme sur `shakespeare_train`
    - pour chaque contexte de `contexts[n]`, prédisez le mot suivant en utilisant la méthode `predict_next`

    Un exemple de sortie possible :
    >>> python mle_ngram_model.py
    n=1
    () --> the

    n=2
    ('King',) --> Philip
    ('I',) --> hope
    ('<s>',) --> Hence

    n=3
    ('<s>', '<s>') --> Come
    ('<s>', 'I') --> would
    ('Something', 'is') --> rotten
    ...
    """
    # Liste de contextes à tester pour n=1, 2, 3
    contexts = {
        1: [()],
        2: [("King",), ("I",), ("<s>",)],
        3: [("<s>", "<s>"), ("<s>", "I"), ("Something", "is"), ("To", "be"), ("O", "Romeo"), ("</s>", "</s>")]
    }

    print("#### test ####")
    print(extract_ngrams_from_sentence(["Alice", "est", "là"], 2))
    print(extract_ngrams([["Alice", "est", "là"], ["Bob", "est", "ici"]], 2))

    counts = count_ngrams([["Alice", "est", "là"], ["Bob", "est", "ici"]], 2)
    print(counts[("est",)]["là"])  # Bigramme connu
    print(counts[("est",)]["Alice"])  # Bigramme inconnu

    mle_counts = compute_MLE(counts)
    print(mle_counts[("est",)]["là"])  # 1/2
    print(mle_counts[("est",)]["Alice"])  # 0/2

    print("#### fin test ####")

    fileName = "shakespeare_train"
    corpus = pre.read_and_preprocess("./data/" + fileName + ".txt")

    for n in contexts.keys():
        Model = NgramModel(corpus, n)
        print("\nn=" + str(n))
        for tuple_n in contexts[n]:
            print("%s --> %s" % (str(tuple_n), str(Model.predict_next(tuple_n))))

    print("--------------")

    for i in range(1,4):
        print("\n--- Pour n = " + str(i) + "---")
        counts = count_ngrams(corpus, i)
        newDict = {}

        for key in counts:
            for k,v in counts[key].items():
                newDict[key + tuple([k])] = v

        sorted_dict = sorted(newDict.items(), key=operator.itemgetter(1), reverse=True)
        count_dict = collections.OrderedDict(sorted_dict)

        for key in list(count_dict)[:20]:
            print(str(key) + " : " + str(count_dict[key]))

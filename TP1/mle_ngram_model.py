"""
Question 1.2.1 à 1.2.7 : implémentation d'un modèle n-gramme MLE.

Les premières fonctions permettent de compter les n-grammes d'un corpus puis de calculer les estimées MLE; vous devrez
les compléter dans les questions 1.2.1 à 1.2.4. Elles sont ensuite encapsulées dans une classe `NgramModel` (déjà écrite),
vous pourrez alors entraîner un modèle n-gramme sur un corpus avec la commande :
>>> lm = NgramModel(corpus, n)

Sauf mention contraire, le paramètre `corpus` désigne une liste de phrases tokenizées
"""
from itertools import chain


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
    pass


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
    pass


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
    pass

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
    pass


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
        pass


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
        3: [("<s>", "<s>"), ("<s>", "I"), ("Something", "is"), ("To", "be"), ("O", "Romeo")]
    }

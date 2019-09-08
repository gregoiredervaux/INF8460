"""
Questions 1.3.1 et 1.3.2 : validation de votre modèle avec NLTK

Dans ce fichier, on va comparer le modèle obtenu dans `mle_ngram_model.py` avec le modèle MLE fourni par NLTK.

Pour préparer les données avant d'utiliser le modèle NLTK, on pourra utiliser
>>> ngrams, words = padded_everygram_pipeline(n, corpus)
>>> vocab = Vocabulary(words, unk_cutoff=k)

Lors de l'initialisation d'un modèle NLTK, il faut passer une variable order qui correspond à l'ordre du modèle n-gramme,
et une variable vocabulary de type Vocabulary.

On peut ensuite entraîner le modèle avec la méthode model.fit(ngrams). Attention, la documentation prête à confusion :
la méthode attends une liste de liste de n-grammes (`list(list(tuple(str)))` et non pas `list(list(str))`).
"""
import nltk
from nltk.lm.models import MLE
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
import mle_ngram_model as mle
import preprocess_corpus as pre


def train_MLE_model(corpus, n):
    """
    Entraîne un modèle de langue n-gramme MLE de NLTK sur le corpus.

    :param corpus: list(list(str)), un corpus tokenizé
    :param n: l'ordre du modèle
    :return: un modèle entraîné
    """
    return mle.NgramModel(corpus, n)


def compare_models(your_model, nltk_model, corpus, n):
    """
    Pour chaque n-gramme du corpus, calcule la probabilité que lui attribuent `your_model`et `nltk_model`, et
    vérifie qu'elles sont égales. Si un n-gramme a une probabilité différente pour les deux modèles, cette fonction
    devra afficher le n-gramme en question suivi de ses deux probabilités.

    À la fin de la comparaison, affiche la proportion de n-grammes qui diffèrent.

    :param your_model: modèle NgramModel entraîné dans le fichier 'mle_ngram_model.py'
    :param nltk_model: modèle nltk.lm.MLE entraîné sur le même corpus dans la fonction 'train_MLE_model'
    :param corpus: list(list(str)), une liste de phrases tokenizées à tester
    :return: float, la proportion de n-grammes incorrects
    """
    n_gram_count = 0
    n_gram_diff = 0
    counts = your_model.counts
    for n_gram_begining in counts:
        for n_gram_end in counts[n_gram_begining]:
            if(counts[n_gram_begining][n_gram_end] != nltk_model.unmasked_score(n_gram_end, n_gram_begining)):
                print("Différente probabilité pour le n_gram " + str(n_gram_begining) +" " +str(n_gram_end))
                print(str(counts[n_gram_begining][n_gram_end]) + " vs " + str(nltk_model.unmasked_score(n_gram_end, n_gram_begining)))
                n_gram_diff += 1
            n_gram_count += 1
    return n_gram_diff/n_gram_count


if __name__ == "__main__":
    """
    Ici, vous devrez valider votre implémentation de `NgramModel` en la comparant avec le modèle NLTK. Pour n=1, 2, 3,
    vous devrez entraîner un modèle nltk `MLE` et un modèle `NgramModel` sur `shakespeare_train`, et utiliser la fonction 
    `compare_models `pour vérifier qu'ils donnent les mêmes résultats. 
    Comme corpus de test, vous choisirez aléatoirement 50 phrases dans `shakespeare_train`.
    """

    fileName = "shakespeare_train"
    corpus = pre.read_and_preprocess("./data/" + fileName + ".txt")
    n = 1
    other_model = nltk.lm.models.MLE(n, corpus)
    my_model = train_MLE_model(corpus, n)
    diff = compare_models(my_model,other_model,corpus,n)
    print(diff)

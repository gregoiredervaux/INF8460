"""
Questions 1.1.6 à 1.1.8 : calcul de différentes statistiques sur un corpus.

Sauf mention contraire, toutes les fonctions renvoient un nombre (int ou float).
Dans toutes les fonctions de ce fichier, le paramètre `corpus` désigne une liste de phrases tokenizées, par exemple :
>>> corpus = [
    ["Alice", "est", "là"],
    ["Bob", "est", "ici"]
]
"""
import preprocess_corpus as pre


def count_tokens(corpus):
    """
    Renvoie le nombre de mots dans le corpus
    """
    nb_token = 0
    for sentence in corpus:
        nb_token += len(sentence)
    return nb_token

def count_types(corpus):
    """
    Renvoie le nombre de types (mots distincts) dans le corpus
    """
    list_type = []
    for sentence in corpus:
        for word in sentence:
            if word not in list_type:
                list_type.append(word)
    return len(list_type)


def get_most_frequent(corpus, n):
    """
    Renvoie les n mots les plus fréquents dans le corpus, ainsi que leurs fréquences

    :return: list(tuple(str, float)), une liste de paires (mot, fréquence)
    """
    dictionnary = {}
    for sentence in corpus:
        for word in sentence:
            if word not in dictionnary:
                dictionnary[word] = 1
            else:
                dictionnary[word] += 1

    sorted_word = []
    sorted_occurence = []
    i = 0

    for key, values in sorted(dictionnary.items(), key=lambda kv: kv[1], reverse=True):
        sorted_word.append(key)
        sorted_occurence.append(dictionnary[key])
        i += 1
        if i >= n: break
    return sorted_word, sorted_occurence

def get_token_type_ratio(corpus):
    """
    Renvoie le ratio nombre de tokens sur nombre de types
    """
    return count_tokens(corpus)/count_types(corpus)


def count_lemmas(corpus):
    """
    Renvoie le nombre de lemmes distincts
    """
    return count_types(pre.lemmatize(corpus))


def count_stems(corpus):
    """
    Renvoie le nombre de racines (stems) distinctes
    """
    return count_types(pre.stem(pre.lemmatize(corpus)))


def explore(corpus):
    """
    Affiche le résultat des différentes fonctions ci-dessus.

    Pour `get_most_frequent`, prenez n=15

    >>> explore(corpus)
    Nombre de tokens: 5678
    Nombre de types: 890
    ...
    Nombre de stems: 650

    """
    print("Nombre de tokens: " + str(count_tokens(corpus)))
    print("Nombre de types: " + str(count_types(corpus)))
    list_freq = get_most_frequent(corpus, 15)
    print("items les plus fréquents: " + str(list_freq[0]))
    print("frequence des items     : " + str(list_freq[1]))
    print("Ration token/type: " + str(get_token_type_ratio(corpus)))
    print("Nombre de lemmes: " + str(count_lemmas(corpus)))
    print("Nombre de stems: " + str(count_stems(corpus)))


if __name__ == "__main__":
    """
    Ici, appelez la fonction `explore` sur `shakespeare_train` et `shakespeare_test`. Quand on exécute le fichier, on 
    doit obtenir :

    >>> python explore_corpus
    -- shakespeare_train --
    Nombre de tokens: 5678
    Nombre de types: 890
    ...

    -- shakespeare_test --
    Nombre de tokens: 78009
    Nombre de types: 709
    ...
    """
    fileName = "shakespeare_train"
    corpus = pre.read_and_preprocess("./data/" + fileName + ".txt")
    print("--" + fileName + "--")
    explore(corpus)

    fileName = "shakespeare_test"
    corpus = pre.read_and_preprocess("./data/" + fileName + ".txt")
    print("--" + fileName + "--")
    explore(corpus)



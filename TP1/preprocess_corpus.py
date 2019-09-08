"""
Questions 1.1.1 à 1.1.5 : prétraitement des données.
"""
import re
import nltk

nltk.download("punkt")
nltk.download("wordnet")

global fileName

def segmentize(raw_text):
    """
    Segmente un texte en phrases.

    >>> raw_corpus = "Alice est là. Bob est ici"
    >>> segmentize(raw_corpus)
    ["Alice est là.", "Bob est ici"]

    :param raw_text: str
    :return: list(str)
    """
    raw_text = re.sub("__URL__", "", raw_text)

    return nltk.sent_tokenize(raw_text)

def tokenize(sentences):
    """
    Tokenize une liste de phrases en mots.

    >>> sentences = ["Alice est là", "Bob est ici"]
    >>> corpus = tokenize(sentences)
    >>> corpus_name
    [
        ["Alice", "est", "là"],
        ["Bob", "est", "ici"]
    ]

    :param sentences: list(str), une liste de phrases
    :return: list(list(str)), une liste de phrases tokenizées
    """

    return [nltk.word_tokenize(str) for str in sentences]

def lemmatize(corpus):
    """
    Lemmatise les mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases lemmatisées
    """
    lemmzer = nltk.WordNetLemmatizer()
    return [[lemmzer.lemmatize(str) for str in list] for list in corpus]


def stem(corpus):
    """
    Retourne la racine (stem) des mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases stemées
    """
    stemmer = nltk.PorterStemmer()
    return [[stemmer.stem(str) for str in list] for list in corpus]

def read_and_preprocess(filename):
    """
    Lit un fichier texte, puis lui applique une segmentation et une tokenization.

    [Cette fonction est déjà complète, on ne vous demande pas de l'écrire]

    :param filename: str, nom du fichier à lire
    :return: list(list(str))
    """
    with open(filename, "r") as f:
        raw_text = f.read()
    return tokenize(segmentize(raw_text))


def test_preprocessing(raw_text, sentence_id=0):
    """
    Applique à `raw_text` les fonctions segmentize, tokenize, lemmatize et stem, puis affiche le résultat de chacune
    de ces fonctions à la phrase d'indice `sentence_id`

    >>> trump = open("data/trump.txt", "r").read()
    >>> test_preprocessing(trump)
    Today we express our deepest gratitude to all those who have served in our armed forces.
    Today,we,express,our,deepest,gratitude,to,all,those,who,have,served,in,our,armed,forces,.
    Today,we,express,our,deepest,gratitude,to,all,those,who,have,served,in,our,armed,force,.
    today,we,express,our,deepest,gratitud,to,all,those,who,have,serv,in,our,arm,forc,.

    :param raw_text: str, un texte à traiter
    :param sentence_id: int, l'indice de la phrase à afficher
    :return: un tuple (sentences, tokens, lemmas, stems) qui contient le résultat des quatre fonctions appliquées à
    tout le corpus
    """
    sentences = segmentize(raw_text)
    print(sentences[sentence_id])
    corpus = tokenize(sentences)
    print(corpus[sentence_id])
    save_corpus_to_txt(corpus, "_mots")
    corpus_lemme = lemmatize(corpus)
    print(corpus_lemme[sentence_id])
    save_corpus_to_txt(corpus, "_lemmes")
    corpus_stem = stem(corpus_lemme)
    print(corpus_stem[sentence_id])
    save_corpus_to_txt(corpus, "_stems")

def save_corpus_to_txt(corpus, suffix):
    with open(fileName + suffix + ".txt", "w") as f:
        for sentence in corpus:
            str = ""
            for word in sentence:
                str += word + " "
            f.write(str + "\n")

if __name__ == "__main__":
    """
    Appliquez la fonction `test_preprocessing` aux corpus `shakespeare_train` et `shakespeare_test`.

    Note : ce bloc de code ne sera exécuté que si vous lancez le script directement avec la commande :
    ```
    python preprocess_corpus.py
    ```
    """
    fileName = "./data/shakespeare_test"
    with open(fileName + ".txt", "r") as f:
        raw_text = f.read()
    test_preprocessing(raw_text)

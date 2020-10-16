import sys
import re
import string
from math import log
from pathlib import Path
from typing import Dict, List, Set, Tuple

import nltk

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

punctuation_regex = re.compile(rf"[{string.punctuation}]")
stopwords = nltk.corpus.stopwords.words("english")


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory: str) -> Dict[str, str]:
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    all_data: Dict[str, str] = {}
    dir_path = Path(directory)
    for file in dir_path.iterdir():
        with file.open("r", encoding="utf-8") as f:
            # Remove first line, which is the URL
            data = "".join(list(f)[1:])
        all_data[file.name] = data
    return all_data


def tokenize(document: str) -> List[str]:
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words: List[str] = []
    for w in nltk.word_tokenize(document):
        wl = w.lower()
        if wl in stopwords:
            continue
        wr = punctuation_regex.sub("", wl)
        if not wr:
            continue
        words.append(wr)
    return words


def compute_idfs(documents: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    num_docs = len(documents)
    all_words = {w for words in documents.values() for w in words}
    idfs = {w: log(num_docs/len({d for d, ws in documents.items() if w in ws}))
            for w in all_words}
    return idfs


def top_files(query: Set[str], files: Dict[str, List[str]], idfs: Dict[str, float], n: int) -> List[str]:
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Represent each file as a tuple[float, str]; the float is the sum of
    #   tf-idf scores and the str is the filename
    file_tf_idfs: List[Tuple[float, str]] = [
        (sum(words.count(w) * idfs.get(w, 0.0) for w in query), file)
        for file, words in files.items()]
    file_tf_idfs.sort(key=lambda x: x[0], reverse=True)
    return [f[1] for f in file_tf_idfs[:n]]


def top_sentences(query: Set[str], sentences: Dict[str, List[str]], idfs: Dict[str, float], n: int) -> List[str]:
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Represent each sentence as a tuple[tuple[float, float], str]; the first
    #   float is the sum of idf scores, the second float is the query term
    #   density, and the str is the sentence
    sent_scores: List[Tuple[Tuple[float, float], str]] = [((
        # The sum of idf scores
        sum(idfs.get(w, 0.0) for w in query if w in words),
        # The query term density
        sum(words.count(w) for w in query)/len(words)
    ), sent)
        for sent, words in sentences.items()]
    sent_scores.sort(key=lambda x: x[0], reverse=True)
    return [f[1] for f in sent_scores[:n]]


if __name__ == "__main__":
    main()

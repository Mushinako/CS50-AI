import re
from re import sub
import sys
from typing import List

import nltk

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> SFull | S SSubConj
SSubConj -> Conj SSub
SSub -> SFull | VP
SFull -> NPP VP
NPP -> NP | NPP PP
NP -> NAdj | Det NAdj
NAdj -> N | Adj NAdj
PP -> P NPP
VP -> VPP | VPP Adv | Adv VPP
VPP -> VN | VPP PP
VN -> V | V NPP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

alphabet_regex = re.compile(r"[A-Za-z]")


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees: List[nltk.Tree] = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence: str) -> List[str]:
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words: List[str] = nltk.word_tokenize(sentence)
    filtered_words = [w.lower() for w in words
                      if alphabet_regex.search(w) is not None]
    return filtered_words


def np_chunk(tree: nltk.Tree) -> List[nltk.Tree]:
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # No need to check for inclusion because it's impossible in this syntax
    np_chunks: List[nltk.Tree] = list(
        tree.subtrees(lambda t: t.label() == "NP"))
    return np_chunks


if __name__ == "__main__":
    main()

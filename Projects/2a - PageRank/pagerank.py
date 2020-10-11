import os
import random
import re
import sys
from collections import Counter
from typing import Dict, Set

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory: str) -> Dict[str, Set[str]]:
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages: Dict[str, Set[str]] = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus: Dict[str, Set[str]], page: str, damping_factor: float) -> Dict[str, float]:
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Check if page has no links
    accessible_pages = corpus[page]
    if not accessible_pages:
        random_prob = 1 / len(corpus)
        return {p: random_prob for p in corpus}

    # Random access
    random_prob = (1 - damping_factor) / len(corpus)
    probs = {p: random_prob for p in corpus}

    # Link access
    link_prob = damping_factor / len(accessible_pages)
    for p in accessible_pages:
        probs[p] += link_prob

    return probs


def sample_pagerank(corpus: Dict[str, Set[str]], damping_factor: float, n: int) -> Dict[str, float]:
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # First choice
    page = random.choice(tuple(corpus.keys()))
    pagerank_choices = []

    for _ in range(n):
        probs = transition_model(corpus, page, damping_factor)
        # Random choice based on the probability
        # `random.choices` always return a list
        page = random.choices(tuple(probs.keys()), tuple(probs.values()))[0]
        pagerank_choices.append(page)

    counter = Counter(pagerank_choices)
    return {p: c/n for p, c in counter.items()}


def iterate_pagerank(corpus: Dict[str, Set[str]], damping_factor: float) -> Dict[str, float]:
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Make the page point to all pages if it points to none
    for page in corpus:
        if not corpus[page]:
            corpus[page] = set(corpus)

    # Initialize iteration conditions
    init_prob = 1 / len(corpus)
    probs = {p: init_prob for p in corpus}

    # (1 - d) / N
    random_prob = (1 - damping_factor) / len(corpus)

    # Iteration
    accuracy = 0.001
    while True:
        # Σ(PR(i)/NumLinks(i))
        link_sum = {p: 0.0 for p in corpus}
        for page in corpus:
            links = corpus[page]
            # PR(i)/NumLinks(i)
            pr_each = probs[page] / len(links)
            for link in links:
                link_sum[link] += pr_each
        # (1 - d) / N + d * Σ
        new_probs = {p: random_prob + damping_factor * link_sum[p]
                     for p in corpus}
        if all(abs(prob-new_prob) < accuracy
               for prob, new_prob in zip(probs.values(), new_probs.values())):
            break
        else:
            probs = new_probs
    return new_probs


if __name__ == "__main__":
    main()

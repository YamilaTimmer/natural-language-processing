
"""
input: .tok bestand met tokens
gebruiker geeft n mee en lengte van nieuwe tekst
documentatie met --help

begin met willekeurige token
vanaf daar trek tokens volgens de berekende waarschijnlijkheid

"""
from collections import Counter
from random import choices
from nlp import filereader
import sys
from tokenize import encoder, decode


def determine_probability(text, tokens, n, len_text, counter):
    counter = Counter(text)
    sequence = []

    # Genereer start tokens, aan de hand van hoe vaak ze voorkomen
    start_tok = choices(list(counter.keys()), weights=list(counter.values()), k=n)

    # Voor unigram
    if n == 1:
        rest_tok = choices(list(counter.keys()), weights=list(counter.values()), k=len_text)

    # Splits tekst op tot ngrams
    ngrams = [tuple(text[i:i + n]) for i in range(len(text) - n)]
    nplus1grams = [tuple(text[i:i + n + 1]) for i in range(len(text) - n)]

    ngram_counts = Counter(ngrams)
    nplus1_counts = Counter(nplus1grams)
    #print(nplus1_counts)

    occurence_dict = {}
    for ngram in ngrams:
        occurence_dict[ngram] = {}
        for nplus1, count in nplus1_counts.items():
            if nplus1[0:n] == ngram:
                occurence_dict[ngram].update({nplus1:count})

    probability_dict = {}
    print(occurence_dict)

    return

def generate_text(counter, n):

    return

def write_output():
    return


def main():
    words = filereader("gutenberg_cancer.txt")
    counter = Counter(words)
    words_tokens, id_to_tok = encoder(words, max_tokens=160, min_freq=20)
    tokens = list(id_to_tok.values())
    determine_probability(words, tokens, 2, 100, counter)


if __name__ == "__main__":

    main()
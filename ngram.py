
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
from tokenize import encoder, decode


def determine_probability(text, tokens, n):
    print(tokens)
    # Splits tekst op tot ngrams
    ngrams = [tuple(text[i:i + n]) for i in range(len(text) - n)]
    nplus1grams = [tuple(text[i:i + n + 1]) for i in range(len(text) - n)]

    # Hoe vaak de ngrams voorkomen en hoe vaak ngrams van één lengte langer voorkomen, om alle woorden te bepalen
    # die andere woorden kunnen opvolgen
    ngram_counts = Counter(ngrams)
    nplus1_counts = Counter(nplus1grams)

    probability_dict = {}

    # tel welke combinaties de ngrams kunnen opvolgen en hoe vaak dit gebeurt
    for ngram in ngrams:
        probability_dict[ngram] = {}
        for nplus1, count in nplus1_counts.items():
            if nplus1[0:n] == ngram:
                total_occurence = ngram_counts[ngram]
                # Voeg woord + kans hierop toe
                probability_dict[ngram].update({nplus1[-1]:count/total_occurence})

    return probability_dict, ngram_counts

def generate_text(n, text, text_len, probability_dict, ngram_counts):
    sequence = []
    counter = Counter(text)


    # Genereer start tokens, aan de hand van hoe vaak ze voorkomen

    # genereer tekst voor unigrams
    if n == 1:
        start_tok = tuple(choices(list(counter.keys()), weights=list(counter.values()), k=n))
        sequence.extend(start_tok)
        rest_tok = choices(list(counter.keys()), weights=list(counter.values()), k=text_len)
        sequence.extend(rest_tok)

    # genereer tekst voor grotere ngrams
    else:
        start_tok = tuple(choices(list(ngram_counts.keys()), weights=list(ngram_counts.values()), k=1))[0]
        sequence.extend(start_tok)

        for i in range(text_len):
            next_word = choices(list(probability_dict[start_tok].keys()), weights=list(probability_dict[start_tok].values()), k=1)[0]
            sequence.append(next_word)
            start_tok = start_tok[1:] + (next_word,)

    return sequence

def write_output(sequence):
    with open("output.txt", "w") as output:
        delimiter = " "

        output.write(delimiter.join(sequence))

    return


def main():
    text = filereader("gutenberg_cancer.txt")
    text = text[1:10000]
    words_tokens, id_to_tok = encoder(text, max_tokens=160, min_freq=20)
    tokens = list(id_to_tok.values())

    n = 3
    probability_dict, ngram_counts = determine_probability(text, tokens, 3)
    sequence = generate_text(3, text, 100, probability_dict, ngram_counts)
    write_output(sequence)


if __name__ == "__main__":

    main()
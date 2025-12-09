
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


def determine_probability(tokens, n):
    # Splits tekst op tot ngrams
    ngrams = [tuple(tuple(tok) for tok in tokens[i:i + n]) for i in range(len(tokens) - n)]
    nplus1grams = [tuple(tuple(tok) for tok in tokens[i:i + n + 1]) for i in range(len(tokens) - n)]

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

def generate_text(n, tokens, text_len, probability_dict, ngram_counts):
    sequence = []

    # genereer tekst voor unigrams
    if n == 1:
        counter = Counter(tokens)

        start_tok = tuple(choices(list(counter.keys()), weights=list(counter.values()), k=n))
        sequence.extend(start_tok)
        rest_tok = choices(list(counter.keys()), weights=list(counter.values()), k=text_len)
        sequence.extend(rest_tok)

    # genereer tekst voor grotere ngrams
    else:
        current_ngram = tuple(choices(list(ngram_counts.keys()), weights=list(ngram_counts.values()), k=1))[0]
        sequence.extend(current_ngram)

        for _ in range(text_len):
            next_word = choices(list(probability_dict[current_ngram].keys()), weights=list(probability_dict[current_ngram].values()), k=1)[0]
            sequence.append(list(next_word))
            current_ngram = current_ngram[1:] + (next_word,)

    return sequence

def write_output(sequence, id_to_tok):
    decoded_sequence = decode(sequence, id_to_tok)

    with open("output.txt", "w") as output:
        output.write(decoded_sequence)

    return


def main():
    text = filereader("gutenberg_cancer.txt")
    text = text[1:10000]
    words_tokens, id_to_tok = encoder(text, max_tokens=160, min_freq=20)

    n = 3
    probability_dict, ngram_counts = determine_probability(words_tokens, 3)
    sequence = generate_text(3, words_tokens, 100, probability_dict, ngram_counts)
    write_output(sequence, id_to_tok)

if __name__ == "__main__":

    main()
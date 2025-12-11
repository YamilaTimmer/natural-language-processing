
"""
input: .tok bestand met tokens
gebruiker geeft n mee en lengte van nieuwe tekst
documentatie met --help

begin met willekeurige token
vanaf daar trek tokens volgens de berekende waarschijnlijkheid

"""
import sys
from collections import Counter, defaultdict
from random import choices
from nlp import filereader

def determine_probability(tokens, n):
    ngram_counts = Counter()
    next_tokens = defaultdict(Counter)

    # Splits tekst op tot ngrams
    for i in range(len(tokens) - n):
        ngram = tuple(tokens[i:i + n])
        next_token = tokens[i + n]
        ngram_counts[ngram] +=1
        next_tokens[ngram][next_token] += 1


    # Hoe vaak de ngrams voorkomen en hoe vaak ngrams van één lengte langer voorkomen, om alle woorden te bepalen
    # die andere woorden kunnen opvolgen
    probability_dict = {}

    for ngram, next_count in next_tokens.items():
        probability_dict[ngram] = {}
        total_occurrence = ngram_counts[ngram]
        for token, count in next_count.items():
            probability_dict[ngram][token] = count / total_occurrence

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
        current_ngram = list(choices(list(ngram_counts.keys()), weights=list(ngram_counts.values()), k=1))[0]
        sequence.append(list(current_ngram))

        for _ in range(text_len):
            next_word = choices(list(probability_dict[current_ngram].keys()), weights=list(probability_dict[current_ngram].values()), k=1)[0]
            sequence.append(list(next_word))
            current_ngram = current_ngram[1:] + (next_word,)

    return sequence

def write_output(sequence, output_file):

    with open(output_file, "w") as output:
        output.write(str(sequence))

    return

def print_help():
    print("""
    N-gram text generator

    Usage:
        python ngram.py <input_file.tok> <n> <length> <output_file>
      
    Parameters:
        <input_file.tok>: tok bestand met tokens, kan worden gegenereerd met behulp van de tokenizer
        <n>: getal dat de lengte van de N-grammen bepaald
        <length>: de lengte van de te genereren tekst
        <output_file>: het pad waar de gegenereerde tekst naartoe geschreven wordt
    
    Options:
      -h, --help   Toon deze helptekst
    """)

def main():
    # --help of --help
    if "-h" in sys.argv or "--help" in sys.argv:
        print_help()
        return

    if len(sys.argv) < 4:
        print("Usage: python ngram.py <input_file.tok> <n> <length> <output_file>")
        return

    tokens = sys.argv[-4]
    n = int(sys.argv[-3])
    text_len = int(sys.argv[-2])
    output_file = sys.argv[-1]

    tokenized_text = filereader(tokens)
    probability_dict, ngram_counts = determine_probability(tokenized_text, n)
    sequence = generate_text(n, tokenized_text, text_len, probability_dict, ngram_counts)
    write_output(sequence, output_file)


if __name__ == "__main__":

    main()
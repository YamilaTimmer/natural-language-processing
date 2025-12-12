"""
N-gram Text Generator

Dit script genereert willekeurige teksten op basis van een N-gram
language model. Het model wordt getraind op een bestand met
tokens (*.tok) en kan zowel unigrams (n=1), bigrams (n=2), trigrams (n=3), enz.
gebruiken om nieuwe tekst te genereren.

Gebruik (command line):
    python ngram.py <input_file1.tok> [<input_file2.tok> ...] -e <file.enc> -n <n> -l <length> -o <output_file>

Parameters:
    <input.tok>    : Input tokenbestand(en), bevatten inputtekst die door de tokenizer is omgezet naar tokens
    <file.enc>     : Bevat de encoding van de tokenizer, met key:value=token:string
    <n>            : Lengte van de n-gram
    <length>       : Lengte van de te genereren tekst
    <output_file>  : Pad waar de gegenereerde tekst wordt opgeslagen

Voorbeeld:
    python ngram.py gutenberg_cancer.tok -e gutenberg_cancer.enc -n 3 -l 100 -o output.txt
"""

import argparse
from collections import Counter, defaultdict
from random import choices
from nlp import filereader, load_enc, decode


def determine_probability(tokens, n):
    """
    Bepaalt hoe waarschijnlijk het is voor bepaalde tokens om ngrammen op te volgen,
    nodig om de 'willekeurige' tekst te kunnen genereren met weights.

    :param tokens: Bevat geëncodeerde versie van de input tekst(en)
    :param n: int dat aangeeft hoe lang de ngrammen zijn
    :return: dict met waarschijnlijkheden voor tokens die volgen na een ngram
    (en hoe vaak deze ngrammen voorkomen)
    """
    ngram_counts = Counter()
    next_tokens = defaultdict(Counter)

    # Splits tekst op tot ngrams
    for i in range(len(tokens) - n):
        ngram = tuple(tokens[i:i + n])
        next_token = tokens[i + n]
        ngram_counts[ngram] +=1
        next_tokens[ngram][next_token] += 1

    # Hoe vaak de ngrams voorkomen en hoe vaak ngrams van één lengte langer voorkomen,
    # om alle woorden te bepalen die andere woorden kunnen opvolgen
    probability_dict = {}

    for ngram, next_count in next_tokens.items():
        probability_dict[ngram] = {}
        total_occurrence = ngram_counts[ngram]
        for token, count in next_count.items():
            probability_dict[ngram][token] = count / total_occurrence

    return probability_dict, ngram_counts

def generate_text(n, tokens, text_len, probability_dict, ngram_counts):
    """
    Genereer willekeurige tekst aan de hand van de waarschijnlijkheden van tokens om ngram
    tokens op te volgen.

    :param n: int dat aangeeft hoe lang de ngrammen zijn
    :param tokens: Bevat geëncodeerde versie van de input tekst(en)
    :param text_len: Gewenste lengte van de te genereren tekst
    :param probability_dict: Dict met waarschijnlijkheden voor tokens die volgen na een ngram
    :param ngram_counts: counter die per ngram bijhoudt hoe vaak deze voorkomt
    :return: de willekeurig gegenereerde sequentie
    """
    sequence = []

    # genereer tekst voor unigrams
    if n == 1:
        counter = Counter(tokens)

        start_tok = choices(
            list(counter.keys()),
            weights=list(counter.values()),
            k=1)[0]

        # genereer nieuwe start als deze niet in prob dict zit (bijv token aan het einde)
        if start_tok not in probability_dict:
            start_tok = choices(
                list(counter.keys()),
                weights=list(counter.values()),
                k=1)[0]

        sequence.extend(start_tok)
        rest_tok = choices(
            list(counter.keys()),
            weights=list(counter.values()),
            k=text_len)

        sequence.extend(rest_tok)
        return sequence

    # genereer tekst voor grotere ngrams (>1)
    else:
        current_ngram = choices(
            list(ngram_counts.keys()),
            weights=list(ngram_counts.values()),
            k=n
        )[0]

        sequence = list(current_ngram)

        for _ in range(text_len - len(current_ngram)):
            # als gekozen ngram niet in prob_dict zit, bijv. als laatste token is gekozen dan
            # zijn er geen opvolgende tokens
            if current_ngram not in probability_dict:
                current_ngram = choices(
                    list(probability_dict.keys()),
                    weights=[ngram_counts[ng] for ng in probability_dict.keys()],
                    k=n
                )[0]

            next_word = choices(
                list(probability_dict[current_ngram].keys()),
                weights=list(probability_dict[current_ngram].values()),
                k=n
            )[0]

            sequence.append(next_word)
            current_ngram = current_ngram[1:] + (next_word,)

        return sequence

def write_output(sequence, output_file):
    """
    Schrijft de output weg naar een bestand

    :param sequence: de willekeurig gegenereerde tekst
    :param output_file: het pad waarnaar de output geschreven moet worden
    """
    with open(output_file, "w", encoding="utf-8") as output:
        output.write("".join(str(tok) for tok in sequence))

def parse_args():
    parser = argparse.ArgumentParser(
        description="N-gram text generator",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "tok_files",
        nargs="+",
        help="Input .tok file(s), meerdere bestanden zijn toegestaan"
    )
    parser.add_argument(
        "-n",
        type=int,
        required=True,
        help="Lengte van de n-grams"
    )
    parser.add_argument(
        "-l", "--length",
        type=int,
        required=True,
        help="Gewenste lengte voor de te genereren tekst"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Outputbestand voor de gegenereerde tekst"
    )
    parser.add_argument(
        "-e", "--enc",
        type=str,
        required=True,
        help="Encodingbestand (.enc) van de tokenizer"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    tokens = args.tok_files
    n = args.n
    text_len = args.length
    output_file = args.output
    enc_file = args.enc

    tokenized_texts = []

    for token_file in tokens:
        tokenized_text = filereader(token_file)
        tokenized_texts.extend(tokenized_text)

    probability_dict, ngram_counts = determine_probability(tokenized_texts, n)
    sequence = generate_text(n, tokenized_texts, text_len, probability_dict, ngram_counts)

    sequence_int = [int(tok) for tok in sequence]

    id_to_tok = load_enc(enc_file)
    sequence_words = [[t] for t in sequence_int]
    decoded_text = decode(sequence_words, id_to_tok)

    write_output(decoded_text, output_file)
    print(f"N-gram tekst met n:{n} en lengte {text_len} succesvol gegenereerd, opgeslagen op {output_file}")

if __name__ == "__main__":
    main()
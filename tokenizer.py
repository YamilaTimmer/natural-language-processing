import sys
import os

# Importeer algemene NLP-functionaliteit
from nlp import filereader, encoder, load_enc, decode


def save_enc(id_to_tok, input_file):
    """
    Sla de BPE-encoding op in een .enc bestand met zelfde naam als gebruikte txt bestand.

    Parameters:
        id_to_tok : dict van token-ID met token-inhoud
        input_file : oorspronkelijke inputbestand, wordt gebruikt om de .enc bestandsnaam te maken
    """
    # Bepaal de map waarin dit script staat
    base = os.path.dirname(os.path.abspath(__file__))
    # Maak de bestandsnaam voor het .enc bestand op basis van de input_file
    filename = os.path.splitext(os.path.basename(input_file))[0] + ".enc"
    path = os.path.join(base, filename)

    with open(path, "w", encoding="utf-8") as f:
        # schrijf token id en bijbehorend token naar bestand
        for k, v in id_to_tok.items():
            f.write(f"{k}:{v}\n")

    print("Encoding saved:", path)


def save_tok(words_tokens, input_file):
    """
    Sla de getokenizeerde woorden op in een .tok bestand.
    """
    # Bepaal de map waarin dit script staat
    base = os.path.dirname(os.path.abspath(__file__))
    # Maak de bestandsnaam voor het .tok bestand op basis van de input_file
    filename = os.path.splitext(os.path.basename(input_file))[0] + ".tok"
    path = os.path.join(base, filename)

    with open(path, "w", encoding="utf-8") as f:
        for w in words_tokens:
            f.write(" ".join(map(str, w)) + "\n")

    print("Tokens saved:", path)


def print_help():
    print("""
Byte-Pair Encoding tokenizer

Usage:
  python tokenizer.py learn <txt_file> [max_tokens] [min_freq]
  python tokenizer.py tokenize <txt_file> <enc_file>
  python tokenizer.py decode <tok_file> <enc_file>

Modes:
  learn
    Leest een .txt tekstbestand in en leert een Byte-Pair Encoding (BPE).
    De encoding wordt opgeslagen in een .enc bestand.

  tokenize
    Zet een .txt bestand om naar tokens met een gegeven .enc bestand.
    Output wordt opgeslagen als .tok.

  decode
    Zet een .tok bestand terug om naar leesbare tekst met behulp van een .enc bestand.

Options:
  -h, --help   Toon deze helptekst
""")


def main():
    # --help of -h
    if "-h" in sys.argv or "--help" in sys.argv:
        print_help()
        return

    if len(sys.argv) < 3:
        print("Usage: python tokenizer.py [learn|tokenize|decode] <input_file>")
        return

    # Modus en inputbestand
    mode = sys.argv[1]
    input_file = sys.argv[2]

    # Default waarden
    max_tokens = 1000
    min_freq = 2

    if mode == "learn":
        if len(sys.argv) > 3:
            max_tokens = int(sys.argv[3])
        if len(sys.argv) > 4:
            min_freq = int(sys.argv[4])

        words = filereader(input_file)
        _, id_to_tok = encoder(words, max_tokens=max_tokens, min_freq=min_freq)
        save_enc(id_to_tok, input_file)
        print(f"BPE learned! Max tokens respected: {len(id_to_tok)}")

    elif mode == "tokenize":
        if len(sys.argv) < 4:
            print("Usage: python tokenizer.py tokenize <txt_file> <enc_file>")
            return

        enc_file = sys.argv[3]
        words = filereader(input_file)
        id_to_tok = load_enc(enc_file)
        tok_to_id = {v: k for k, v in id_to_tok.items()}

        words_tokens = []
        for w in words:
            i = 0
            w_tok = []
            while i < len(w):
                match = None
                for l in range(len(w) - i, 0, -1):
                    sub = w[i:i + l]
                    if sub in tok_to_id:
                        match = sub
                        break
                if match:
                    w_tok.append(tok_to_id[match])
                    i += len(match)
                else:
                    w_tok.append(tok_to_id[w[i]])
                    i += 1
            words_tokens.append(w_tok)

        save_tok(words_tokens, input_file)


    elif mode == "decode":
        if len(sys.argv) < 4:
            print("Usage: python tokenizer.py decode <tok_file> <enc_file>")
            return

        enc_file = sys.argv[3]
        id_to_tok = load_enc(enc_file)

        tokens_list = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens_list.append([int(t) for t in line.strip().split()])

        text = decode(tokens_list, id_to_tok)

        base = os.path.dirname(os.path.abspath(__file__))
        out_file = os.path.splitext(os.path.basename(input_file))[0] + "_decoded.txt"
        path = os.path.join(base, out_file)

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        print("Decoded text saved:", path)


if __name__ == "__main__":
    main()

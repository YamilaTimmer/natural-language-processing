import sys
from collections import Counter
import os

def filereader(file_path):
    """
    Lees een tekstbestand in en splits het in een lijst van woorden.
    Parameters:
        file_path: pad naar het tekstbestand dat ingelezen moet worden.

    Returns:
        list: lijst met woorden uit het bestand, gescheiden door witruimte.
    """
    # Open en lees het bestand
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Verwijder witruimte aan het begin/eind en splitst de tekst in woorden
    return text.strip().split()


def get_pairs(words_tokens):
    """
    Genereer een lijst van alle token-paren in de woordenlijst.

    Parameters:
        words_tokens : lijst van woorden, waarbij elk woord een lijst van tokens is.

    Returns:
        pairs: lijst van tuples, elk tuple is een paar opeenvolgende tokens.
    """
    pairs = []

    # Loop door elk woord in de lijst
    for w in words_tokens:
        # Voeg elk paar opeenvolgende tokens toe
        for i in range(len(w)-1):
            pairs.append((w[i], w[i+1]))

    return pairs


def encoder(words, max_tokens=1000, min_freq=2):
    """
    Maak een BPE, Byte-Pair Encoding, op basis van de gegeven woorden.
    Parameters:
        words : lijst met woorden uit de input tekst.
    max_tokens : maximale aantal unieke tokens voor de encoding, user kan dit kiezen met default 1000
    min_freq : hoe vaak een paar tokens ten minste moet voorkomen om te worden samengevoegd tot een nieuw token.
    user kan dit kiezen met default 2

    Returns:
        words_tokens : lijst van woorden, elk woord is een lijst van token-ID's
        id_to_tok : dict met token-ID's als keys en token-strings als values
    """

    # Controleer of max_tokens minimaal het aantal unieke letters bevat. Geef waarschuwing en gebruik het minimaal
    # aantal unieke letters als max_tokens
    unique_chars = set(c for w in words for c in w)
    if max_tokens < len(unique_chars):
        print(f"Warning: max_tokens ({max_tokens}) is smaller than number of unique letters ({len(unique_chars)}). "
              f"Setting max_tokens to {len(unique_chars)}.")
        max_tokens = len(unique_chars)

    tok_dict = {}
    counter = 1
    words_tokens = []

    # Initialiseer single-character tokens
    for w in words:
        w_tok = []
        for c in w:
            if c not in tok_dict:
                tok_dict[c] = counter
                counter += 1
            w_tok.append(tok_dict[c])
        words_tokens.append(w_tok)

    # Omgekeerde mapping: ID -> token
    id_to_tok = {v: k for k, v in tok_dict.items()}

    # Merge paren totdat max_tokens of min_freq bereikt is
    while True:
        if len(id_to_tok) >= max_tokens:
            break

        pairs = Counter(get_pairs(words_tokens))
        if not pairs:
            break
        top_pair, freq = pairs.most_common(1)[0]
        if freq < min_freq:
            break

        if len(id_to_tok) + 1 > max_tokens:
            break

        # Voeg nieuwe token toe
        new_tok = id_to_tok[top_pair[0]] + id_to_tok[top_pair[1]]
        new_id = max(id_to_tok.keys()) + 1
        id_to_tok[new_id] = new_tok

        # Update woordenlijst met nieuwe token
        new_words_tokens = []
        for w in words_tokens:
            i = 0
            new_w = []
            while i < len(w):
                if i < len(w)-1 and (w[i], w[i+1]) == top_pair:
                    new_w.append(new_id)
                    i += 2
                else:
                    new_w.append(w[i])
                    i += 1
            new_words_tokens.append(new_w)
        words_tokens = new_words_tokens

    return words_tokens, id_to_tok


def save_enc(id_to_tok, input_file):
    """
    Sla de BPE-encoding op in een .enc bestand met zelfde naam als gebruikte txt bestand.

    Parameters:
        id_to_tok : dict van token-ID met token-inhoudt
        input_file : oorspronkelijke inputbestand, wordt gebruikt om de .enc bestandsnaam te maken

    Returns:
        path: pad naar het opgeslagen .enc bestand
    """
    # Bepaal de map waarin dit script staat
    base = os.path.dirname(os.path.abspath(__file__))
    # Maak de bestandsnaam voor het .enc bestand op basis van de input_file
    filename = os.path.splitext(os.path.basename(input_file))[0] + ".enc"
    path = os.path.join(base, filename)
    # Open het bestand in schrijfmodus en sla de encoding op
    with open(path, "w", encoding="utf-8") as f:
        #schrijf token id en bijbehorend token naar bestand
        for k, v in id_to_tok.items():
            f.write(f"{k}:{v}\n")
    print("Encoding saved:", path)
    return path


def save_tok(words_tokens, input_file):
    """
    Sla de getokenizeerde woorden op in een .tok bestand.

    Parameters:
        words_tokens : ijst van woorden met token-ID's
    input_file : oorspronkelijke inputbestand, wordt gebruikt om de .tok bestandsnaam te maken

    Returns:
        path: pad naar het opgeslagen .tok bestand
    """
    # Bepaal de map waarin dit script staat
    base = os.path.dirname(os.path.abspath(__file__))
    # Maak de bestandsnaam voor het .tok bestand op basis van de input_file
    filename = os.path.splitext(os.path.basename(input_file))[0] + ".tok"
    path = os.path.join(base, filename)
    # Open het bestand in schrijfmodus en sla de tokenizatie op
    with open(path, "w", encoding="utf-8") as f:
        for w in words_tokens:
            f.write(" ".join(map(str, w)) + "\n")
    print("Tokens saved:", path)
    return path


def load_enc(enc_file):
    """
    Laad een .enc bestand en maak een mapping van token-ID naar token-inhoud.

    Parameters:
        enc_file : pad naar het .enc bestand

    Returns:
        id_to_tok: dict van token-ID:token-inhoud
    """
    id_to_tok = {}
    with open(enc_file, "r", encoding="utf-8") as f:
        for line in f:
            # Verwijder witruimte aan het begin/eind en splits de regel op ':' in key en value
            k, v = line.strip().split(":", 1)
            # Voeg het token-ID en de bijbehorende token-inhoud toe aan de dictionary
            id_to_tok[int(k)] = v
    return id_to_tok


def decode(tokens_list, id_to_tok):
    """
    Zet een lijst van token-ID's terug om naar tekst.

    Parameters:
        tokens_list : lijst van woorden met token-ID's
        id_to_tok : dict van token-ID:token-inhoud

    Returns:
        De gedecodeerde tekst
    """
    words = []
    for w in tokens_list:
        words.append("".join(id_to_tok[t] for t in w))
    return " ".join(words)


def main():
    # --help of learn --help
    if "-h" in sys.argv or "--help" in sys.argv:
        print_help()
        return
    if len(sys.argv) < 3:
        print("Usage: python tokenize_.py [learn|tokenize|decode] <input_file> [enc_file] [max_tokens] [min_freq]")
        return

    # Het eerste argument bepaalt de modus: learn, tokenize of decode
    mode = sys.argv[1]
    # Het tweede argument is het pad naar het inputbestand
    input_file = sys.argv[2]

    # default waarden
    max_tokens = 1000
    min_freq = 2

    if mode == "learn":
        if len(sys.argv) > 3:
            max_tokens = int(sys.argv[3])
        if len(sys.argv) > 4:
            min_freq = int(sys.argv[4])

    # Als we in "learn"-modus zitten, lees optionele argumenten voor max_tokens en min_freq
    if mode == "learn":
        words = filereader(input_file)
        _, id_to_tok = encoder(words, max_tokens=max_tokens, min_freq=min_freq)
        save_enc(id_to_tok, input_file)
        print(f"BPE learned! Max tokens respected: {len(id_to_tok)}")


    elif mode == "tokenize":
        # Controleer of de gebruiker een .enc bestand heeft opgegeven
        if len(sys.argv) < 4:
            print("Usage: python tokenize_.py tokenize <txt_file> <enc_file>")
            return
        # Het pad naar het .enc bestand
        enc_file = sys.argv[3]
        words = filereader(input_file)
        # Laad de BPE-encoding mapping van token-ID naar token-string
        id_to_tok = load_enc(enc_file)
        # Maak een omgekeerde mapping: token-string -> token-ID
        tok_to_id = {v: k for k, v in id_to_tok.items()}

        words_tokens = []
        # Tokeniseer elk woord afzonderlijk
        for w in words:
            i = 0
            w_tok = []
            while i < len(w):
                match = None
                # Zoek het langste token dat past vanaf positie i
                for l in range(len(w)-i, 0, -1):
                    sub = w[i:i+l]
                    if sub in tok_to_id:
                        match = sub
                        break
                if match:
                    # Voeg de token-ID toe en verschuif de index
                    w_tok.append(tok_to_id[match])
                    i += len(match)
                else:
                    # 1 karakter als token
                    w_tok.append(tok_to_id[w[i]])
                    i += 1
            words_tokens.append(w_tok)
        save_tok(words_tokens, input_file)
        print("Tokenization done!")


    elif mode == "decode":
        # Controleer of de gebruiker een .enc bestand heeft opgegeven
        if len(sys.argv) < 4:
            print("Usage: python tokenize_.py decode <tok_file> <enc_file>")
            return
        # Het pad naar het .enc bestand
        enc_file = sys.argv[3]
        # Laad de BPE-encoding mapping van token-ID naar token-string
        id_to_tok = load_enc(enc_file)

        tokens_list = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                # Zet de spatie-gescheiden token-ID's om naar een lijst van integers
                tokens_list.append([int(t) for t in line.strip().split()])

        # Decodeer de lijst van token-ID's terug naar tekst
        text = decode(tokens_list, id_to_tok)

        # Bepaal de map waarin dit script staat
        base = os.path.dirname(os.path.abspath(__file__))

        # Maak de bestandsnaam voor het gedecodeerde tekstbestand
        out_file = os.path.splitext(os.path.basename(input_file))[0] + "_decoded.txt"
        path = os.path.join(base, out_file)

        # Schrijf de gedecodeerde tekst naar het bestand
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        # Print een bevestiging voor de gebruiker
        print("Decoded text saved:", path)

def print_help():
    print("""
Byte-Pair Encoding tokenizer

Usage:
  python tokenize.py learn <txt_file> [max_tokens (default = 1000)] [min_freq (default = 2)]
  python tokenize.py tokenize <txt_file> <enc_file>
  python tokenize.py decode <tok_file> <enc_file>

Modes:
  learn
    Leest een .txt tekstbestand in en leert een Byte-Pair Encoding (BPE).
    De encoding wordt opgeslagen in een .enc bestand met zelfde naam als txt bestand.

    Optional arguments:
      max_tokens   maximaal aantal tokens dat mag worden aangemaakt (default: 1000)
      min_freq     minimale frequentie van een token-paar om samen te voegen (default: 2)

  tokenize
    Leest een .txt bestand en zet het om naar tokens met een gegeven .enc bestand.
    Het resultaat wordt opgeslagen in een .tok bestand met zelfde naam als txt en enc bestand.

  decode
    Leest een .tok bestand en zet dit terug om naar leesbare tekst
    met behulp van een .enc bestand.
    Het resultaat wordt opgeslagen als _decoded.txt met zelfde naam als tok en enc bestand.

Options:
  -h, --help   Toon deze helptekst
""")

if __name__ == "__main__":
    main()

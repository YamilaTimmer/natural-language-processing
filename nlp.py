from collections import Counter

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
        for i in range(len(w) - 1):
            pairs.append((w[i], w[i + 1]))

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
        print(
            f"Warning: max_tokens ({max_tokens}) is smaller than number of unique letters ({len(unique_chars)}). "
            f"Setting max_tokens to {len(unique_chars)}."
        )
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
                if i < len(w) - 1 and (w[i], w[i + 1]) == top_pair:
                    new_w.append(new_id)
                    i += 2
                else:
                    new_w.append(w[i])
                    i += 1
            new_words_tokens.append(new_w)
        words_tokens = new_words_tokens

    return words_tokens, id_to_tok


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

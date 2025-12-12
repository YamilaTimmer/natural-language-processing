from collections import Counter
import pandas as pd

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
    
def load_tok_file(tok_file):
    """Lees een .tok bestand (lijst van token-ID's)

    Parameters:
        tok_file : pad naar het .tok bestand

    Returns:
        tokenized_data : lijst van token-ID's
    """
    tokenized_data = []
    with open(tok_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tokenized_data.append(list(map(int, line.split())))
    return tokenized_data

def file_merger(list_of_files):
    """
    Deze functie zet een lijst met filepaths om naar gezamenlijke lijst met met woorden en de hoeveelheid woorder per bestand.
    
    param: list_of_files [string filepaths]
    
    return: final_list_of_words [merged list of words of all files]
    return: list_of_len_per_file [int len of words per file]
    """
    list_of_words_per_file = []
    list_of_len_per_file = []
    final_list_of_words = []

    for file_index in range(len(list_of_files)):
        words = filereader(list_of_files[file_index])
        list_of_words_per_file.append(words)
        len_of_doc = len(words)
        list_of_len_per_file.append(len_of_doc)

    for list_index in range(len(list_of_words_per_file)):
        current_word_list = list_of_words_per_file[list_index]
        for word_index in range(len(current_word_list)):
            current_word = current_word_list[word_index]
            final_list_of_words.append(current_word)

    return final_list_of_words, list_of_len_per_file


def group_encoder(max_tokens,min_freq,list_of_words,list_of_doc_len):
    """
    Deze functie gebruikt de encoder functie op alle bestanden zodat hier een gezamenlijk tokenizatie op word toegepast.
    en splitst de lijst met woorden terug naar de lijsten met woorden per bestand, maar dan getokeniseerd
    
    Param: max tokens, maximale hoeveelheid tokens dat gegenereerd kan worden
    Param: min freq, minimale freqwentie dat nodig is om een token te defineren
    Param: list_of_words, voledige lijst met woorden
    Param: list_of_doc_len, lijst met de lengte van de hoeveelheid woorden per bestand

    return: uncoupled_token_list_per_doc, lijst[document] van lijsten[woorden in tokens]
    return groupt_token_dict, key = token, value = woorden/letters
    """
    uncoupled_token_lists_per_doc = []

    groupt_list_of_words, groupt_token_dict = encoder(list_of_words,max_tokens,min_freq)
    start = 0
    stop = 0
    # uncoupeling based on old len per doc
    for len_index in range(len(list_of_doc_len)):
        current_len = list_of_doc_len[len_index]-1
        stop = current_len + start
        words_in_file = groupt_list_of_words[start:stop]
        uncoupled_token_lists_per_doc.append(words_in_file)
        start = stop+1

    return uncoupled_token_lists_per_doc, groupt_token_dict


def multi_hot_encoding(token_lists,tokens_dict,list_of_names):
    """
    Deze functie genereerd een multi hot encoding dataframe

    """    
    def word_checker(current_token_lists,key):
        """
        deze functie checkt of de key in de lijst met tokenlijsten zit
        Return: boolean
        """
        result = False
        for word_index in range(len(current_token_lists)):
            current_word_in_tokens = current_token_lists[word_index]
            if key in current_word_in_tokens:
                result = True
            else:
                continue

        return result
    
    data = {}
    row_list = []
    
    for file_index in range(len(list_of_names)):
        file_name = list_of_names[file_index]
        data[file_name] = []

    for key in tokens_dict.keys():
        row_list.append(key)
        for file_index in range(len(list_of_names)):
            current_file = list_of_names[file_index]
            current_token_list = token_lists[file_index]
            bool_check = word_checker(current_token_list,key)
            if bool_check == True:
                old_list = data.get(current_file)
                old_list.append(1)
                data[current_file] = old_list
            else:
                old_list = data.get(current_file)
                old_list.append(0)
                data[current_file] = old_list

    df = pd.DataFrame(data, index=row_list)


    return df


def frequency_checker(token_lists,tokens_dict,list_of_names,result_type):
    """
    Deze functie telt de hoeveelheid keren dat een token voorkomt in een bestand
    """
    def word_counter(current_token_lists,key):
        """   
        Telt de hoeveelheid dat de key in de lijst van tokens voorkomt
        """
        count = 0
        for word_index in range(len(current_token_lists)):
            current_word_in_tokens = current_token_lists[word_index]
            if key in current_word_in_tokens:
                count+=1
            else:
                continue
        return count
    

    
    data = {}
    total_token_counter = 0

    for file_index in range(len(list_of_names)):
        file_name = list_of_names[file_index]
        data[file_name] = []
    
    row_list = []
    for key in tokens_dict.keys():
        row_list.append(key)
        for file_index in range(len(list_of_names)):
            current_file = list_of_names[file_index]
            current_token_list = token_lists[file_index]
            word_count = word_counter(current_token_list,key)
            total_token_counter += word_count
            old_list = data.get(current_file)
            old_list.append(word_count)
            data[current_file] = old_list

    df = pd.DataFrame(data,index=row_list)
    if result_type == "frac":
        df = df / total_token_counter

    if result_type == "perc":
        df = df/total_token_counter*100

    return df

def build_token_mappings(enc):
    """Maak dicts van token:index en index:token
    Parameters:
        enc : lijst van token-ID's

    Returns:
        all_tokens : lijst van token-ID's
        token_to_idx : lijst van token-ID's
        idx_to_token : lijst van token-ID's
    """
    all_tokens = list(enc.values())
    token_to_idx = {tok: i for i, tok in enumerate(all_tokens)}
    idx_to_token = {i: tok for i, tok in enumerate(all_tokens)}
    return all_tokens, token_to_idx, idx_to_token

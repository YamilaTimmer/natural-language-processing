import os
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import argparse
from nlp import load_tok_file, load_enc, build_token_mappings

# Waarschuwing van sklearn niet weergeven aan gebruiker.
# Bijvoorbeeld als er niet veel oefendata is, komt er een waarschuwing dat het model niet volledig convergeert.
warnings.filterwarnings("ignore", category=UserWarning)


def build_dataset(tokenized_data, enc, token_to_idx, n=2):
    """
    Bouwt een dataset voor het trainen van een MLP (multi-layer perceptron) op basis van context rondom
    een target token
    Parameters:
        tokenized_data: lijsten van token-ID's
        enc: dict van token-id:token
        token_to_idx : dict van token:token-index
        n: grootte van context window (aan beide kanten van target token) met dfault 2

    Returns:
        X: np array, matrix met one-hot gecodeerde contextfeatures voor elk trainingsvoorbeeld
        Y: np array met target labels (token-ID's verschoven naar 0-based indexering).
        token_counter: frequentie van elk targettoken binnen de dataset
    """
    X, Y = [], []
    num_tokens = len(token_to_idx)
    token_counter = Counter()

    for seq in tokenized_data:
        # Sequenties die te kort zijn voor een context van n links en n rechts overslaan
        if len(seq) < 2*n + 1:
            continue

        # Doorloop tokens die een volledige context hebben
        for i in range(n, len(seq) - n):
            # Verzamel contexttokens links en rechts van de target
            context_ids = seq[i-n:i] + seq[i+1:i+n+1]
            # One-hot vector aanmaken op basis van alle mogelijke tokens
            x_vec = np.zeros(num_tokens)
            # Voor elke context-token de juiste index op 1 zetten
            for t_id in context_ids:

                # ID → tokenstring
                if t_id not in enc:
                    continue  # veiligheidscheck

                tok = enc[t_id]  # bijvoorbeeld "er", "##ing", "c"

                # Alleen verwerken als token bestaat in token_to_idx
                if tok not in token_to_idx:
                    continue

                idx = token_to_idx[tok]
                x_vec[idx] = 1

                # Target ID → target tokenstring
            target_id = seq[i]
            if target_id not in enc:
                continue

            target_tok = enc[target_id]

            # Target moet ook bestaan in mapping
            if target_tok not in token_to_idx:
                continue

            # sklearn labels moeten ints zijn 0..N-1
            Y.append(token_to_idx[target_tok])
            X.append(x_vec)

            token_counter[target_tok] += 1
    return np.array(X), np.array(Y), token_counter


def train_mlp(X, Y, hidden_size):
    """
    Traint een MLP (multi-layer perceptron) op basis van de gegenereerde dataset.

    Parameters:
        X: np array, matrix met one-hot gecodeerde contextfeatures voor elk trainingsvoorbeeld.
        Y: np array met target labels (token-ID's verschoven naar 0-based indexering).
        hidden_size: aantal neuronen in verborgen laag (door user te kiezen) default is 50

    Returns:
        mlp: MLPClassifier, getrainde model
    """
    # hidden_layer_sizes bepaalt hoeveel neuronen er in de verborgen laag zitten (default is 50)
    # max_iter is het maximaal aantal trainingsiteraties die het netwerk mag hebben (500 is goed voor kleine datasets)
    # learning_rate_init is de beginsnelheid waarmee het netwerk leert (simpele embeddings op kleine dataset 0.01)
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_size,),
                        max_iter=500,
                        learning_rate_init=0.01)
    mlp.fit(X, Y)
    return mlp


def save_embeddings_txt(mlp, token_to_idx, output_file):
    """
    Embeddings opslaan in txt bestand. Embeddings zijn de gewichten tussen de inputlaag en de verborgenlaag van de MLP.

    Parameters:
        mlp: MLPClassifier, getrainde model
        token_to_idx: dict van token:inputvector index
        output_file: bestandsnaam voor output file

    Returns:
        embeddings: np array met embeddings
    """
    embeddings = mlp.coefs_[0]  # input -> hidden gewichten

    with open(output_file, 'w', encoding='utf-8') as f:
        for tok, idx in token_to_idx.items():
            vec = " ".join(map(str, embeddings[idx]))
            f.write(f"{tok} {vec}\n")
    print(f"Embeddings opgeslagen in {output_file}")
    return embeddings


def plot_embeddings(embeddings, token_to_idx, min_len=0):
    """
    Plotten van embeddings in 2D, eerste 2 dimensies.

    Parameters:
        embeddings: embeddingsmatrix uit getrainde model
        token_to_idx: dict van token:inputvector index
        min_len: minimale tokenlengte om weer te geven, default 0 zodat alles geplot wordt.
                 Kan door user meegegeven worden als je bijvoorbeeld alleen langere tokens in de plot wilt hebben.
    """
    if embeddings.shape[1] < 2:
        print("Te weinig dimensies voor 2D-plot")
        return

    plt.figure(figsize=(12, 12))
    for tok, idx in token_to_idx.items():
        if len(tok) < min_len:
            continue

        coord = embeddings[idx][:2]
        plt.scatter(coord[0], coord[1], color='blue')
        plt.annotate(tok, (coord[0], coord[1]), fontsize=8)
    plt.title(f"Token Embeddings (min_len={min_len})")
    plt.xlabel("Dimensie 1")
    plt.ylabel("Dimensie 2")
    plt.grid(True)
    plt.show()


def parse_arguments():
    """
    Parseert command-line argumenten met argparse.
    """
    parser = argparse.ArgumentParser(
        description="Train een MLP om token embeddings te leren."
    )

    # Verplichte argumenten
    parser.add_argument("tok_file", help="Pad naar het .tok bestand")
    parser.add_argument("enc_file", help="Pad naar het .enc bestand")

    # Optionele argumenten
    parser.add_argument("--window", type=int, default=2,
                        help="Context window size n (default: 2)")
    parser.add_argument("--hidden", type=int, default=50,
                        help="Aantal neuronen in de verborgen laag (default: 50)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot de embeddings in 2D")
    parser.add_argument("--minlen", type=int, default=0,
                        help="Minimale tokenlengte voor de plot (default: 0)")

    return parser.parse_args()


def main():
    """
    Hoofdprogramma: laadt de data, bouwt dataset, traint MLP en slaat embeddings op.
    Optioneel plotten via --plot.
    """
    args = parse_arguments()

    # Data en mappings laden
    tokenized_data = load_tok_file(args.tok_file)
    enc = load_enc(args.enc_file)
    all_tokens, token_to_idx, idx_to_token = build_token_mappings(enc)

    # Dataset bouwen
    X, Y, token_counter = build_dataset(tokenized_data, enc, token_to_idx, args.window)
    if X.size == 0:
        print("Dataset te klein voor de gegeven window size.")
        return

    # Trainen van het model
    mlp = train_mlp(X, Y, args.hidden)

    # Embeddings opslaan
    emb_file = os.path.splitext(args.tok_file)[0] + ".emb"
    embeddings = save_embeddings_txt(mlp, token_to_idx, emb_file)

    # Plotten indien gevraagd
    if args.plot:
        plot_embeddings(embeddings, token_to_idx, min_len=args.minlen)


if __name__ == "__main__":
    main()

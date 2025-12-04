from collections import defaultdict
import sys

def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("Fout: bestand niet gevonden.")
        sys.exit(1)
    except Exception as e:
        print(f"Fout: {e}")
        sys.exit(1)

def learn_bpe(corpus, num_merges=10):
    vocab = defaultdict(int)
    for sentence in corpus.split('.'):
        words = sentence.strip().split()
        for word in words:
            chars = ['<'] + list(word) + ['>']
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                vocab[pair] += 1

    merges = []
    for _ in range(num_merges):
        if not vocab:
            break
        most_frequent = max(vocab, key=vocab.get)
        merges.append(most_frequent)

        new_vocab = defaultdict(int)
        for pair, count in vocab.items():
            if pair != most_frequent:
                new_vocab[pair] += count
        vocab = new_vocab

    return merges

def apply_bpe(text, merges):
    chars = ['<'] + list(text) + ['>']
    for merge in reversed(merges):
        merged = ''.join(merge)
        new_chars = []
        i = 0
        while i < len(chars) - 1:
            if (chars[i], chars[i + 1]) == merge:
                new_chars.append(merged)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        if i < len(chars):
            new_chars.append(chars[-1])
        chars = new_chars
    return chars

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gebruik: py tokenize.py <bestand.txt> [aantal_merges als getal(optioneel, default = 10)]")
        sys.exit(1)

    filename = sys.argv[1]

    if len(sys.argv) >= 3:
        try:
            num_merges = int(sys.argv[2])
        except ValueError:
            print("Aantal merges moet een geheel getal zijn.")
            sys.exit(1)
    else:
        num_merges = 10  # default

    corpus = read_file(filename)
    merges = learn_bpe(corpus, num_merges=num_merges)
    print("Learned merges:", merges)

    example = "tomaat"
    print(f"BPE voor '{example}':", apply_bpe(example, merges))

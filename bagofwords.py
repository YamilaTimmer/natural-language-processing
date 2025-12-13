"""
Bag of Words Model

Het Bag of Words model houdt geen rekening met de volgorde van de tokens en beschouwt de bestanden als een ongeordende
hoeveelheid tokens/woorden. Wel houdt het rekening met hoeveel tokens er zijn en hoevaak deze voorkomen in de inputbestanden.
Dit script genereert BoW outputbestanden voor de inputbestanden, met daarin voor elke token een getal dat aangeeft hoe veel
de token voorkomt per inputbestand. Dit kan worden gedaan als absolute waarde met {-t freq -c count} maar ook met 0-en en 1-en
die aangeven of een token wel/niet voorkomt in een document.

Gebruik (command line):
    python bagofwords.py <input_file1.txt> [<input_file2.txt> ...] -t {multi/freq/tfidf} -m <max_tokens> -f <min_freq> -c {count/frac/perc}

Parameters:
    <input.txt>            : Input tesktbestand(en),
    <multi/freq/tfidf>     : 3 verschillende manieren van encoding:
        - multi-hot encoding (multi) = Geeft een reeks 0-en en 1-en voor tokens die respectievelijk wel/niet voorkomen in een document
        - frequency encoding = Codeert voor elk voorkomend token de frequentie (als aantal, als fractie of als percentage).
            Het resultaat is een reeks integers of getallen van 0.0 tot 1.0 of 0 tot 100.
        - Term-Frequency Inverse-Document-Frequency (TF-IDF) = telt voor elk token hoe vaak dit voorkomt in elk afzonderlijk
            document en berekent dan een getalwaarde, kent een hogere 'score' toe aan tekens die veel voorkomen in één document,
            en dat nauwelijks voorkomt in andere documenten.
    <max_tokens>           : Maximale hoeveelheid tokens dat gegenereerd kan worden
    <min_freq>             : Hoe vaak een combinatie van tekens moet voorkomen om omgezet te worden tot token
    <count/frac/perc>      : Alleen bij -t freq, geeft aan of output als fractie/percentage gegeven moet worden
        - count= geeft terug hoe vaak alle tokens voorkomen als absolute waarde
        - frac= geeft terug hoe vaak alle tokens voorkomen als fractie
        - perc= geeft terug hoe vaak alle tokens voorkomen als percentage

Voorbeeld:
    python bagofwords.py input_file1.txt input_file2.txt -t freq/ -m 1000 -f 4 -c perc

"""
from nlp import file_merger,group_encoder,multi_hot_encoding,frequency_checker,tf_idf_calc
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Bag of Words analyser",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-m", "--max_tokens",
        type=int,
        default=1000,
        help="Maximale hoeveelheid tokens dat gegenereerd kan worden"
    )
    parser.add_argument(
        "-f","--min_freq",
        type=int,
        default=2,
        help="Minimale hoeveelheid dat nodig is voor een token om gegenereerd te worden"
    )
    parser.add_argument(
        "-t","--type_of_bag",
        choices=["multi", "freq", "tfidf"],
        type=str,
        required=True,
        help="type van bag_of_words resultaten"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="input txt files"
        )
    parser.add_argument("-c","--count_type",
                        type=str,
                        choices=["count", "frac", "perc"],
                        default="count",
                        help="type frequency analyse in getal fractie of percentage default is getal")

    return parser.parse_args()

def df_printer(df):
    """
    Writes dataframe to txt

    param: pandas df
    """
    with open("BoW_results.bow","w") as writer:
        writer.write(df.to_string())
    print("Output written to 'BoW_results.bow'")

def main():
    args = parse_args()
    max_tokens = args.max_tokens
    min_freq = args.min_freq
    files = args.input_files
    type_o_b = args.type_of_bag
    ct = args.count_type
    
    merged_words, len_of_files = file_merger(files)
    uncoupled_token_list_of_lists, token_dict = group_encoder(max_tokens,min_freq,merged_words,len_of_files)

    df = ""

    if type_o_b == "multi":
        df = multi_hot_encoding(uncoupled_token_list_of_lists,token_dict,files)

    if type_o_b == "freq":
        if ct == "frac":
            df = frequency_checker(uncoupled_token_list_of_lists,token_dict,files,ct)
        elif ct == "perc":
            df = frequency_checker(uncoupled_token_list_of_lists,token_dict,files,ct)
        else:
            df = frequency_checker(uncoupled_token_list_of_lists,token_dict,files," ")

    if type_o_b == "tfidf":
        df = tf_idf_calc(uncoupled_token_list_of_lists,token_dict,files)

    df_printer(df)

if __name__ == "__main__":
    main()
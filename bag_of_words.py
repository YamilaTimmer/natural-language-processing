"""


"""
from nlp import file_merger,group_encoder,multi_hot_encoding,frequency_checker,tf_idf_calc
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Bag of Words analyser",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-m", "--max_tokens",
        type=int,
        required=True,
        help="maximale hoeveelheid tokens dat gegeneerd kan worden"
    )
    parser.add_argument(
        "-f","--min_freq",
        type=int,
        required=True,
        help="Minimale hoeveelheid dat nodig is voor een token om gegenereerd te worden"
    )

    parser.add_argument(
        "-t","--type_of_bag",
        type=str,
        required=True,
        help="type van bag_of_words resultaten : multi,freq,tfidf"
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="input txt files"
        )

    parser.add_argument("-c","--count_type",
                        type=str,
                        required=False,
                        help="type frequency analyse in getal fractie of percentage default is getal optioneel argumenten zijn: frac,perc")

    return parser.parse_args()

def df_printer(df):
    """Writes dataframe to txt
    param: pandas df
    """
    with open("BoW_results.bow","a") as writer:
        writer.write(df.to_string())
    return

def main():
    args = parse_args()
    max_tokens = args.max_tokens
    min_freq = args.min_freq
    files = args.input_files
    type_o_b = args.type_of_bag
    ct = args.count_type
    
    merged_words, len_of_files = file_merger(files)
    uncoupled_token_list_of_lists, token_dict = group_encoder(max_tokens,min_freq,merged_words,len_of_files)

    if type_o_b == "multi":
        df = multi_hot_encoding(uncoupled_token_list_of_lists,token_dict,files)
    
    if type_o_b == "freq":
        if ct == "frac":
            df = frequency_checker(uncoupled_token_list_of_lists,token_dict,files,ct)
        if ct == "perc":
            df = frequency_checker(uncoupled_token_list_of_lists,token_dict,files,ct)
        else:
            df = frequency_checker(uncoupled_token_list_of_lists,token_dict,files," ")

    if type_o_b == "tfidf":
        df = tf_idf_calc(uncoupled_token_list_of_lists,token_dict,files)

    df_printer(df)

    return

if __name__ is "__main__":
    main()
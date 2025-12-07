import pandas as pd
import numpy as np
import sys


def filereader(inputfile):
    data = []
    with open(inputfile,"r") as file:
        for line in file:
            stript_line = line.strip()
            data.append(stript_line.split(" "))
        
    return_list = []
    for input_list in range(len(data)):
        data_list = data[input_list]
        for word in range(len(data_list)):
            return_list.append(data_list[word])

    return return_list

def encoder(word_list):
    token_dict = {}
    token_value_dict = {}
    counter = 1

    # start vocab
    for word_index in range(len(word_list)):
        current_word = word_list[word_index]
        for letter_index in range(len(current_word)):
            letter = current_word[letter_index]
            if token_value_dict.get(letter) == None:
                token_value_dict[letter] = 1
                token_dict[letter] = counter
                counter += 1
            else:
                current_value = token_value_dict.get(letter)
                new_value = current_value + 1
                token_value_dict[letter] = new_value
    # tokenize original words 1 toker per letter
    words_in_tokens = []
    for word_index in range(len(word_list)):
        current_word = word_list[word_index]
        word_in_tokens = []
        for letter_index in range(len(current_word)):
            current_letter = current_word[letter_index]
            new_token = token_dict.get(current_letter)
            word_in_tokens.append(new_token)
        words_in_tokens.append(word_in_tokens)
    # reverse key value
    reversed_token_dict = {}
    for key,value in token_dict.items():
        new_key = value
        new_value = key
        reversed_token_dict[new_key] = new_value


    def pair_merger(token_lists, tokens_dict):
        new_token_lists = []
        temp_dict = {}
        pair_dict = {}
        for token_list_index in range(len(token_lists)):
            current_token_list = token_lists[token_list_index]
            if len(current_token_list) < 2:
                continue
            else:
                end_token_index = 1
                token_list_range = len(current_token_list)-1
                for token_index in range(token_list_range):
                    #print(current_token_list)
                    first_token_string = tokens_dict.get(current_token_list[token_index])
                    first_token = current_token_list[token_index]
                    secon_token_string = tokens_dict.get(current_token_list[end_token_index])
                    second_token = current_token_list[end_token_index]
                    end_token_index += 1

                    candidate = first_token_string + secon_token_string
                    if temp_dict.get(candidate) == None:
                        temp_dict[candidate] = 1
                        pair_dict[candidate] = [first_token,second_token]
                    else:
                        current_value = temp_dict.get(candidate)
                        new_value = current_value + 1
                        temp_dict[candidate] = new_value

        key_list = []
        value_list = []
        for key,value in temp_dict.items():
            key_list.append(key)
            value_list.append(value)

        higest_value_index = value_list.index(max(value_list))
        new_string_key = key_list[higest_value_index]

        old_key_list = []
        for key in tokens_dict.keys():
            old_key_list.append(key)

        new_token_key = max(old_key_list)+1
        tokens_dict[new_token_key] = new_string_key
        new_pair = pair_dict.get(new_string_key)
        
        for token_list_index in range(len(token_lists)):
            current_token_list = token_lists[token_list_index]
            temp_token_list = current_token_list.copy()
            end_index = 1
            for token_index in range(len(current_token_list)-1):
                first_token = current_token_list[token_index]
                second_token = current_token_list[end_index]

                if first_token is new_pair[0] and second_token is new_pair[1]:
                    temp_token_list[token_index] = new_token_key
                    temp_token_list[end_index] = None
                    end_index += 1
                else:
                    end_index += 1
            # remove nonetypes
            if None in (temp_token_list):
                temp_token_list.remove(None)
                while None in (temp_token_list):
                    temp_token_list.remove(None)
            new_token_lists.append(temp_token_list)

        return new_token_lists, tokens_dict

    result_token_list = words_in_tokens.copy()
    result_token_dict = reversed_token_dict.copy()
    max_len = 10
    # condense all words into tokens
    while max_len > 2:
        result_token_list, result_token_dict = pair_merger(result_token_list,reversed_token_dict)
        temp = []
        for i in range(len(result_token_list)):
            current_token_list = result_token_list[i]
            temp.append(len(current_token_list))
        
        max_len = max(temp)
    

    return result_token_list,result_token_dict

def Encwriter(token_dict):
    with open("result_encoding.enc","w") as writer:
        for key,value in token_dict.items():
            line = str(key)+":"+str(value)+"\n"
            writer.write(line)
    return

def Tokwriter(token_list):
    line = ""
    for token_index in range(len(token_list)):
        line+= str(token_list[token_index])+" "
    with open("result_tokens.tok", "w") as writer:
        writer.write(line)
    return

def main():
    # TODO modify arguments for max tokens and min pair frequency
    if sys.argv[1] == "enc":
        path = sys.argv[2]
        token_list = filereader(path)
        new_tokens, enc_dict = encoder(token_list)
        Encwriter(enc_dict)
    
    if sys.argv[1] == "enc_tok":
        path = sys.argv[2]
        token_list = filereader(path)
        new_tokens, enc_dict = encoder(token_list)
        Encwriter(enc_dict)
        Tokwriter(new_tokens)

    if sys.argv[1] == "translate":
        token_path = sys.argv[2]
        encoding_path = sys.argv[3]
        # add translater function

    return

#main()
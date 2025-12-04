import pandas as pd
import numpy as np



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

def encoder(word_list, merge_attempts, min_merge_value):
    def merge(word_list, merger_len, mergen_min_req):
        merger_candidates = {}
        
        for word_index in range(len(word_list)):
            current_word = word_list[word_index]
            if len(current_word) < merger_len:
                continue
            else: 
                for letter_index in range(len(current_word)):
                    if letter_index > merger_len:
                        continue
                    else:
                        end_index = letter_index + merger_len
                        candidate = current_word[letter_index: end_index]
                        if merger_candidates.get(candidate) == None:
                            merger_candidates[candidate] = 1
                        else:
                            current_value = merger_candidates.get(candidate)
                            new_value = current_value + 1
                            merger_candidates[candidate] = new_value

        candidate_list = []
        for candidate, value in merger_candidates.items():
            if value >= mergen_min_req:
                candidate_list.append(candidate)
            else:
                continue
        
        return candidate_list
    
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


    for merge_attempt in range(merge_attempts):
        merge_len = merge_attempt + 2
        candidate_list = merge(word_list, merge_len, min_merge_value)
        for candidate_index in range(len(candidate_list)):
            new_candidate = candidate_list[candidate_index]
            token_dict[new_candidate] = counter
            counter += 1
    
    #resut =  merge(word_list,2,5)
    #print(resut)
    #print(token_dict)
    #print(token_value_dict)

    return token_dict


def main():
    path = "test.txt"
    filereader(path)

    return

#main()
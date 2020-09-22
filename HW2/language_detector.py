import argparse
import os, re
import random
import collections
import math


def preprocess(line):
    # DO NOT CHANGE THIS METHOD unless you are done with bigrams and you are trying to get extra credit

    # get rid of the stuff at the end of the line (spaces, tabs, new line, etc.)
    line = line.rstrip()
    # lower case
    line = line.lower()
    # remove everything except characters and white space
    line = re.sub("[^a-z ]", '', line)
    # tokenized, not done "properly" but sufficient for now
    tokens = line.split()

    # update this when working with trigrams (need '$$')
    # you can also change the preprocessing (keep numbers, do not lower case, etc.)
    tokens = ['$'+token+'$' for token in tokens]
    
    return tokens


def create_model(path):
    # This is just some Python magic ...
    # unigrams will return 0 if the key doesn't exist
    unigrams = collections.defaultdict(int)
    # and then you have to figure out what bigrams will return
    bigrams = collections.defaultdict(lambda: collections.defaultdict(int))

    f = open(path, 'r',  encoding="utf8")
    ## You shouldn't visit a token more than once

    for l in f.readlines():
        tokens = preprocess(l)
        if len(tokens) == 0:
            continue
        
        for token in tokens:
            # FIXME Update the counts for unigrams and bigrams
            for i in range(1,len(token)-1): # I can't figure out why we need $, I guess to avoid the last char error?!?
                curr_char = token[i]
                next_char = token[i+1]
                unigrams[curr_char] += 1
                bigrams[curr_char][next_char] += 1

          
            

    # FIXME After calculating the counts, calculate the smoothed log probabilities
    distinct_unigrams = len(unigrams) # should be 26 but who knows?
    unigrams_count = sum(unigrams.values())
    smoothed_bigrams_probs = collections.defaultdict(lambda: collections.defaultdict(int))

    # smoothed log probabilities
    for key in bigrams:
        key_occur = (sum(bigrams[key].values()))
        for value in bigrams[key]:
            smoothed_bigrams_probs[key][value] = math.log((bigrams[key][value] + 1) /(key_occur + distinct_unigrams)) #add-one smoothing

    # return the actual model: bigram (smoothed log) probabilities and unigram counts (the latter to smooth
    # unseen bigrams in predict(...)
    return {"bigrams":smoothed_bigrams_probs, "unigrams_count": unigrams_count }


def predict(file, model_en, model_es):
    prediction = None

    # FIXME Use the model to make predictions.
    # FIXME: Predict whichever language gives you the highest (smoothed log) probability
    # - remember to do exactly the same preprocessing you did when creating the model (that's what it is a method)
    # - you may want to use an additional method to calculate the probablity of a text given a model (and call it twice)
    f = open(file, 'r',  encoding="utf8")
    en_total_prob = 0
    es_total_prob = 0
    for l in f.readlines():
        tokens = preprocess(l)
        if len(tokens) == 0:
            continue
        for token in tokens:
            for i in range(1,len(token)-1): 
                curr_char = token[i]
                next_char = token[i+1]
                en_curr_prob = model_en['bigrams'][curr_char][next_char]
                es_curr_prob = model_es['bigrams'][curr_char][next_char]
                if (en_curr_prob == 0): # if probs is 0, use smoothing
                    en_curr_prob = math.log(1/ (model_en['unigrams_count']+26))
                if (es_curr_prob == 0): # if probs is 0, use smoothing
                    es_curr_prob = math.log(1/ (model_es['unigrams_count']+26))
                en_total_prob += en_curr_prob
                es_total_prob += es_curr_prob

    if (en_total_prob > es_total_prob):
        prediction = 'English'
    else:
        prediction = 'Spanish'
    
    # prediction should be either 'English' or 'Spanish'
    return prediction


def main(en_tr, es_tr, folder_te):
    # DO NOT CHANGE THIS METHOD

    # STEP 1: create a model for English with en_tr
    model_en = create_model(en_tr)

    # STEP 2: create a model for Spanish with es_tr
    model_es = create_model(es_tr)

    # STEP 3: loop through all the files in folder_te and print prediction
    folder = os.path.join(folder_te, "en")
    print("Prediction for English documents in test:")
    for f in os.listdir(folder):
        f_path =  os.path.join(folder, f)
        print(f"{f}\t{predict(f_path, model_en, model_es)}")
    
    folder = os.path.join(folder_te, "es")
    print("\nPrediction for Spanish documents in test:")
    for f in os.listdir(folder):
        f_path =  os.path.join(folder, f)
        print(f"{f}\t{predict(f_path, model_en, model_es)}")


if __name__ == "__main__":
    # DO NOT CHANGE THIS CODE

    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_TR_EN",
                        help="Path to file with English training files")
    parser.add_argument("PATH_TR_ES",
                        help="Path to file with Spanish training files")
    parser.add_argument("PATH_TEST",
                        help="Path to folder with test files")
    args = parser.parse_args()

    main(args.PATH_TR_EN, args.PATH_TR_ES, args.PATH_TEST)

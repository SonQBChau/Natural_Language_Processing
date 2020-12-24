import argparse
import os, re
import random
import collections
import math


def preprocess(line):
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
    tokens = ['$$'+token+'$$' for token in tokens]
    
    return tokens


def create_model(path):
    # This is just some Python magic ...
    # unigrams will return 0 if the key doesn't exist
    unigrams = collections.defaultdict(int)
    # and then you have to figure out what bigrams will return
    bigrams = collections.defaultdict(lambda: collections.defaultdict(int))
    trigrams = collections.defaultdict(lambda: collections.defaultdict(int))

    f = open(path, 'r',  encoding="utf8")
    ## You shouldn't visit a token more than once
    for l in f.readlines():
        tokens = preprocess(l)
        if len(tokens) == 0:
            continue
        for token in tokens:
            # FIXME Update the counts for unigrams and bigrams
            for i in range(2,len(token)-2): 
                char_1 = token[i]
                char_2 = token[i+1]
                char_3 = token[i+2]
                unigrams[char_1] += 1
                bigrams[char_1][char_2] += 1
                trigrams[(char_1,char_2)][char_3] += 1
          
           

    #  After calculating the counts, calculate the smoothed log probabilities
    unigrams_count = float(sum(unigrams.values()))

    # raw probabilities of unigram, bigram, trigram
    unigram_raw_probs = collections.defaultdict(int)
    for char_1 in unigrams:
        unigram_raw_probs[char_1] = unigrams[char_1] / unigrams_count

    bigram_raw_probs = collections.defaultdict(lambda: collections.defaultdict(int))
    for char_1 in bigrams:
        char_1_occur = float(sum(bigrams[char_1].values()))
        for char_2 in bigrams[char_1]:
            bigram_raw_probs[char_1][char_2] = bigrams[char_1][char_2] / char_1_occur

    trigram_raw_probs = collections.defaultdict(lambda: collections.defaultdict(int))
    trigram_lerp_probs = collections.defaultdict(lambda: collections.defaultdict(int))

    # lambda1+2+3 = 1, lambda is trained on held out corpus for maximum probability
    lambda1= 0.3
    lambda2= 0.4
    lambda3= 0.3
    for char_1_char_2 in trigrams:
        char_1_char_2_occur = float(sum(trigrams[char_1_char_2].values()))
        char_1 = char_1_char_2[0]
        char_2 = char_1_char_2[1]
        for char_3 in trigrams[char_1_char_2]:
            trigram_raw_probs[char_1_char_2][char_3] = trigrams[char_1_char_2][char_3] / char_1_char_2_occur
            # simple interpolation
            # P(wi|wi−1,wi−2) = λ3PML(wi|wi−1,wi−2) + λ2PML(wi|wi−1) + λ1PML(wi)
            trigram_lerp_probs[char_1_char_2][char_3] = lambda1 * unigram_raw_probs[char_1]  + lambda2 * bigram_raw_probs[char_1][char_2]  + lambda3 * trigram_raw_probs[char_1_char_2][char_3]
    

    return trigram_lerp_probs


def predict(file, model_en, model_es):
    prediction = None

    #  Use the model to make predictions.
    # Predict whichever language gives you the highest (smoothed log) probability
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
            for i in range(2,len(token)-2): 
                char_1 = token[i]
                char_2 = token[i+1]
                char_3 = token[i+2]
                en_curr_prob = model_en[char_1,char_2][char_3]
                es_curr_prob = model_es[char_1,char_2][char_3]
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

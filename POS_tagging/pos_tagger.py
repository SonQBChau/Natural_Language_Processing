import argparse
import collections
import math
import operator
import random
import utils
import numpy as np 


def create_model(sentences):
    prior_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    priors = collections.defaultdict(lambda: collections.defaultdict(float))
    likelihood_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    likelihoods = collections.defaultdict(lambda: collections.defaultdict(float))
    majority_tag_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    majority_baseline = collections.defaultdict(str)
    tag_counts = collections.defaultdict(int)
    for sentence in sentences:
        for i, token in enumerate(sentence):
            tag_counts[token.tag] += 1
            # I used the first <s> so I uncommented the skip check below
            # if i == 0:
            #     continue
            majority_tag_counts[sentence[i].word][sentence[i].tag] += 1

            # likelihood count
            likelihood_counts[token.tag][token.word] += 1

    for word in majority_tag_counts:
        majority_baseline[word] = max(majority_tag_counts[word].items(), key=operator.itemgetter(1))[0]
    
    # likelihood
    for tag in likelihood_counts:
        tag_occur = (sum(likelihood_counts[tag].values()))
        for word in likelihood_counts[tag]:
            likelihoods[tag][word] = likelihood_counts[tag][word] /tag_occur  

    ############################
    # Calculate prior and likelihood probabilities (after getting the prior and likelihood counts.
    # You can modify the data structures above to fit your needs, you choose what works best for you
    ############################
    
    # prior counts
    unigrams_tag = collections.defaultdict(int)
    for sentence in sentences:
        for i in range(0,len(sentence)-1):
            curr_tok = sentence[i]
            next_tok = sentence[i+1]
            unigrams_tag[curr_tok.tag] += 1
            prior_counts[curr_tok.tag][next_tok.tag] += 1
   
    # prior probabilities
    for key in prior_counts:
        key_occur = (sum(prior_counts[key].values()))
        # for value in prior_counts[key]:
        for value in unigrams_tag:
            priors[key][value] = (prior_counts[key][value] + 1) /(key_occur + len(unigrams_tag)) #add-one smoothing
   
  

    return priors, likelihoods, majority_baseline, tag_counts


def predict_tags(sentences, model, mode='always_NN'):
    assert mode in ['always_NN', 'majority', 'hmm']

    priors, likelihoods, majority_baseline, tag_counts = model
    for sentence in sentences:
    # for sentence in sentences[0:1000]:
        if mode == 'always_NN':
            for token in sentence:
                token.tag = "NN"
        elif mode == 'majority':
            for token in sentence:
                token.tag = majority_baseline[token.word]
        elif mode == 'hmm':
            ############################
            # 1. Create the Viterbi Matrix (one per sentence)
            # 2. Fill the Viterbi Matrix
            # 3. Reconstruct the optimal sequence of tags
            # (The code below just assigns random tags)
            ############################
            tags = [tag for tag  in tag_counts]
            # get the optimal sequence from viterbi
            optimal_pos = viterbi(sentence,tags, priors, likelihoods)
            for token in sentence:
                token.tag = optimal_pos[token.word]
                      
        else:
            assert False

    return sentences

############################
# for i in 1...T: 
#   viterbi[t,1] = P(w1|t) x P(t|<s>)
# for w in 2...W:
#   for t in 1...T:
#       ......
############################
def viterbi(sentence, tags, priors, likelihoods):
    viterbi =  collections.defaultdict(lambda: collections.defaultdict(float))
    optimal_pos = collections.defaultdict(str)
    T = len(tags) 
    W = len(sentence) 
    
    best_first_prob = -1
    best_first_tag = ''
    # find viterbi for first column of the matrix
    for t in range(1 , T): 
        tag = tags[t]
        word = sentence[1].word
        # calculate the probability based on likelihoods and priors
        viterbi[word][tag] = (likelihoods[tag][word] * priors[tags[0]] [tag], tag)

        # this block to find the highest probability tag
        if (viterbi[word][tag][0] > best_first_prob):
            best_first_prob = viterbi[word][tag][0]
            best_first_tag = tag
    optimal_pos[word] = best_first_tag
    
    # find viterbi for the rest of the matrix
    for w in range (2, W):
        word = sentence[w].word
        prior_word = sentence[w-1].word
        best_tag = ''
        best_word_prob = -1
        for t in range(1 , T): 
            best_prob = -1
            best_prev_tag = -1
            for j in range(1 , T): 
                tag = tags[j]
                # calculate the probability based on likelihoods and priors and previous viterbi
                current_prob = viterbi[prior_word][tag][0] * likelihoods[tags[t]][word] * priors [tag][tags[t]]

                # this block to find the highest probability
                if (current_prob > best_prob):
                    best_prob = current_prob
                    best_prev_tag = tag
            viterbi[word][tags[t]] = (best_prob, best_prev_tag)

            # this block to find the highest probability tag
            if (best_prob > best_word_prob):
                best_word_prob = best_prob
                best_tag = tags[t]

        # reconstruct the optimal sequence of tags
        optimal_pos[word]=best_tag
    
    # this is for debug
    # print(viterbi)
    # print(optimal_pos)

    return optimal_pos


if __name__ == "__main__":
    ############################
    ## DO NOT MODIFY THIS METHOD
    ############################

    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_TR",
                        help="Path to train file with POS annotations")
    parser.add_argument("PATH_TE",
                        help="Path to test file (POS tags only used for evaluation)")
    args = parser.parse_args()

    tr_sents = utils.read_tokens(args.PATH_TR) #, max_sents=1)
    # test=True ensures that you do not have access to the gold tags (and inadvertently use them)
    te_sents = utils.read_tokens(args.PATH_TE, test=True)

    model = create_model(tr_sents)

    gold_sents = utils.read_tokens(args.PATH_TR)
    for mode in ['always_NN', 'majority', 'hmm']:
        predictions = predict_tags(tr_sents, model, mode=mode)
        accuracy = utils.calc_accuracy(gold_sents, predictions)
        print(f"Accuracy in train {'('+mode+')':12}"
              f"[{len(list(gold_sents))} sentences]: {accuracy:6.2f} [not that useful, mostly a sanity check]")
    print()

    # read sentences again because predict_tags(...) rewrites the tags
    gold_sents = utils.read_tokens(args.PATH_TE)
    for mode in ['always_NN', 'majority', 'hmm']:
        predictions = predict_tags(te_sents, model, mode=mode)
        accuracy = utils.calc_accuracy(gold_sents, predictions)
        print(f"Accuracy in test {'('+mode+')':12}"
              f"[{len(list(gold_sents))} sentences]: {accuracy:6.2f}")

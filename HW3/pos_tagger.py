import argparse
import collections
import math
import operator
import random

import utils


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
            if i == 0:
                continue
            majority_tag_counts[sentence[i].word][sentence[i].tag] += 1

    for word in majority_tag_counts:
        majority_baseline[word] = max(majority_tag_counts[word].items(), key=operator.itemgetter(1))[0]

    ############################
    
    # YOUR CODE GOES HERE
    # Calculate prior and likelihood probabilities (after getting the prior and likelihood counts.
    # You can modify the data structures above to fit your needs, you choose what works best for you
    ############################
    unigrams_tag = collections.defaultdict(int)
    for sentence in sentences:
        for i in range(0,len(sentence)-1):
            curr_tok = sentence[i]
            next_tok = sentence[i+1]
            unigrams_tag[curr_tok.tag] += 1
            prior_counts[curr_tok.tag][next_tok.tag] += 1
  

    # smoothed log probabilities
    for key in prior_counts:
        key_occur = (sum(prior_counts[key].values()))
        for value in prior_counts[key]:
            priors[key][value] = math.log((prior_counts[key][value] + 1) /(key_occur + len(unigrams_tag))) #add-one smoothing
    
  

    return priors, likelihoods, majority_baseline, tag_counts


def predict_tags(sentences, model, mode='always_NN'):
    assert mode in ['always_NN', 'majority', 'hmm']

    priors, likelihoods, majority_baseline, tag_counts = model
    for sentence in sentences:
        if mode == 'always_NN':
            for token in sentence:
                token.tag = "NN"
        elif mode == 'majority':
            for token in sentence:
                token.tag = majority_baseline[token.word]
        elif mode == 'hmm':
            ############################
            # YOUR CODE GOES HERE
            # 1. Create the Viterbi Matrix (one per sentence)
            # 2. Fill the Viterbi Matrix
            # 3. Reconstruct the optimal sequence of tags
            # (The code below just assigns random tags)
            ############################
            for token in sentence:
                token.tag = random.choice(list(tag_counts.keys()))
        else:
            assert False

    return sentences


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

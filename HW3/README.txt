
I recommend you use line 71:
    for sentence in sentences[0:1000]:
and run only 1000. It would take too long to run all. 

I implemented: 
- Calculate the prior and likelihood probabilities (create model): ok
– Create and fill the Viterbi Matrix correctly (predict tags): ok
– Obtain the optimal sequence of tags after filling the Viterbi Matrix (predict tags): I get the optimal sequence based on the highest probability. However, this give me the low accuracy on the test set and I couldn't figure why
– Your strategies to deal with unknown words and the report: I didn't get to this part yet.

How to run: 
python pos_tagger.py ../data/train ../data/heldout
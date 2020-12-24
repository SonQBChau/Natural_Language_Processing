import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")
# note: put text file in the same folder
book = open("57886-0.txt", encoding='utf-8').read()
doc = nlp(book)

num_of_sents = len(list(doc.sents))
print('Number of sentences: % s'% num_of_sents)

num_of_toks = len(doc)
print('Number of tokens: % s'% num_of_toks)

avg_num_tk = num_of_toks// num_of_sents
print('Average number of tokens per sentence: % s'% avg_num_tk)

pos_list = [ token.pos_ for token in doc]
uniq_pos = len(set(pos_list))
print('Number of unique part-of-speech tags.: % s'% uniq_pos)

countered_pos_list = Counter(pos_list)
print ('The counts of the most frequent part-of-speech tags: % s'% countered_pos_list.most_common(5))
print ('The counts of the least frequent part-of-speech tags: % s'% countered_pos_list.most_common()[:-5-1:-1])

label_list = [ ent.label_ for ent in doc.ents]
countered_label_list = Counter(label_list)
print('The number of named entities of each type:')
print (countered_label_list.most_common())


sent_list = doc.sents
neg_counter = 0
for sent in sent_list:
    res = [token.text for token in sent.subtree if token.dep_ == 'neg']

    if bool(res):
        neg_counter = neg_counter + 1

print ('The number of sentences with at least one neg syntactic dependency: % s'% neg_counter)

sent_list = doc.sents
went_counter = 0
for sent in sent_list:
    if sent.root.text == 'went':
        went_counter = went_counter + 1
        

print ('The number of sentences whose root (in the dependency tree) is the verb went: % s'% went_counter)
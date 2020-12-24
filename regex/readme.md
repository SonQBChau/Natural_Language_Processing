##### pos_spacy: calculate the following
1. Number of sentences.
2. Number of tokens.
3. Average number of tokens per sentence.
4. Number of unique part-of-speech tags.
5. The counts of the most frequent part-of-speech tags (the top 5).
6. The counts of the least frequent part-of-speech tags (the bottom 5).
7. The number of named entities of each type (how many persons, gpes, etc. are there?).
8. The number of sentences with at least one neg syntactic dependency.
9. The number of sentences whose root (in the dependency tree) is the verb went.

##### verb23rdperson:
$ python3 verb23rdperson.py verbs.txt

| Verb      | Verb 3rd |
| -------| ------|
|kiss    |kisses|
|fix     |fixes|
|go      |goes|
|watch   |watches|
|crash   |crashes|
|go      |goes|
|carry   |carries|
|hurry   |hurries|
|study   |studies|
|deny    |denies|
|run     |runs|
|smile   |smiles|
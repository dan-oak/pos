# Simple English part-of-speech tagger

In corpus linguistics, part-of-speech tagging (POS tagging or POST), also called grammatical tagging or word-category disambiguation, is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context—i.e., its relationship with adjacent and related words in a phrase, sentence, or paragraph. A simplified form of this is commonly taught to school-age children, in the identification of words as nouns, verbs, adjectives, adverbs, etc. [Wikipedia]

Written in Python 3. Dependencies: `scipy`, `numpy`, `nltk`.   
_NB. all of the above dependencies are not crucial, but still used in code - they will be removed later._  

## Usage
_Note_: use Python 3 (i.e. `python3` instead of `python` on some systems)
    
### Interactive sentence tagger
```
python postag-interactive.py
```

### Awesome statistics
```
python correctness.py data/Brown_tagged_dev.txt output/Brown_tagged_dev.txt
```

### Example

    Sentence > Hello, part-of-speech tagger! NLP is an exciting field of applicable science!
    Tagged : Hello/PRT ,/. part-of-speech/NOUN tagger/NOUN !/. NLP/NOUN is/VERB an/DET exciting/ADJ field/NOUN of/ADP applicable/ADJ science/NOUN !/.

## Presentation

Checkout my conference presentation on the project in `misc/presentation.pdf`.
Althought it's in ukrainian there are cool images! ;)

![Hidden Markov Model](/misc/screenshot-2.png)
![Exupéry quote](/misc/screenshot-1.png)

## Perspective

There is still work to do: documentation, code refinement, model improvement etc. Currently I'm using only hidden markov model. 

Glad to collaborate on this and similar projects!

Regards,  
Dahn.

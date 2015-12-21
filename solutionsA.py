import math
import nltk
import time
from collections import Counter
from itertools import chain

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with
# tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples
# expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    tokens = [sent.strip().split()+[STOP_SYMBOL] for sent in training_corpus]
    unigrams = [(token,) for sent in tokens for token in sent]
    unigram_c = Counter(unigrams)
    l = len(unigrams)
    unigram_p = {k: math.log(float(v),2)-math.log(l,2) for k,v in unigram_c.items()}
    tokens = [[START_SYMBOL]+s for s in tokens]
    bigrams = [t for S in tokens for t in nltk.bigrams(S)]
    bigram_c = Counter(bigrams)
    bigram_p = {}
    for k,v in bigram_c.items():
        p = math.log(float(v),2)
        w = k[:1]
        if w == (START_SYMBOL,):
            p -= math.log(float(l),2)
        else:
            p -= math.log(float(unigram_c.get(w)),2)
        bigram_p[k] = p
    tokens = [[START_SYMBOL]+s for s in tokens]
    trigrams = [t for S in tokens for t in nltk.trigrams(S)]
    trigram_c = Counter(trigrams)
    trigram_p = {}
    for k,v in trigram_c.items():
        p = math.log(float(v),2)
        w = k[:2]
        if w == (START_SYMBOL,)*2:
            p -= math.log(float(l),2)
        else:
            p -= math.log(float(bigram_c.get(w)),2)
        trigram_p[k] = p
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the
# ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  +
            ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] +
            ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens
# separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is
# the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    def ngramswitch(n, sentence):
        S = sentence.split()+[STOP_SYMBOL]
        if n == 1:
            return[(token,) for token in S]
        S = [START_SYMBOL]+S
        if n == 2:
            return nltk.bigrams(S)
        S = [START_SYMBOL]+S[:-2]
        return nltk.trigrams(S)
    scores = []
    for sentence in corpus:
        ngrams = ngramswitch(n,sentence)
        s = 0
        for ngram in ngrams:
            if not (ngram in ngram_p):
                s = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            else:
                s += ngram_p.get(ngram)
        scores.append(s)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# Calculates scores (log probabilities) for every sentence with a linearly
# interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that
# express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    for sentence in corpus:
        tokens = [START_SYMBOL]*2+sentence.split()+[STOP_SYMBOL]
        sentence_scores = []
        for trigram in nltk.trigrams(tokens):
            bigram = trigram[1:]
            unigram = bigram[1:]
            la = 3
            triscore = trigrams.get(trigram, MINUS_INFINITY_SENTENCE_LOG_PROB)
            if triscore == MINUS_INFINITY_SENTENCE_LOG_PROB:
                p = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            biscore = bigrams.get(bigram, MINUS_INFINITY_SENTENCE_LOG_PROB)
            uniscore = unigrams.get(unigram, MINUS_INFINITY_SENTENCE_LOG_PROB)
            if MINUS_INFINITY_SENTENCE_LOG_PROB in [triscore,biscore,uniscore]:
                sentence_scores = [MINUS_INFINITY_SENTENCE_LOG_PROB]
                break
            trigram = 2.0**triscore
            bigram = 2.0**biscore
            unigram = 2.0**uniscore
            if biscore == MINUS_INFINITY_SENTENCE_LOG_PROB:
                la -= 1
                bigram = 0.0
            if uniscore == MINUS_INFINITY_SENTENCE_LOG_PROB:
                la -= 1
                unigram = 0.0
            sentence_scores.append(math.log((trigram + bigram + unigram)/float(la),2))
        scores.append(sum(sentence_scores))
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    start_time = time.time()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    end_time = time.time()

    # print total time to run Part A
    print "Part A time: " + str(end_time - start_time) + ' sec'

    globals().update(locals())

if __name__ == "__main__": main()

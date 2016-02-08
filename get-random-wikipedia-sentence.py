import wikipedia as wi, nltk, main, pickle


# TODO: consider using https://en.wikipedia.org/wiki/Special:Random
# https://en.wikipedia.org/wiki/Special:RandomInCategory/Mathematics

# FIXME: this selects only uncategorized articles :<
" titles "
t = wi.random(pages=5)

print('Random wikipedia articles: "' + '", "'.join(t) + '"')

z = nltk.data.load('tokenizers/punkt/english.pickle')

def c(s):
    return 10 < len(s) < 17

""" model parameters """
p = {}
for n in 'e_values known_words q_values tagset'.split():
    with open('objects/' + n + '.pkl', 'rb') as f:
        p[n] = pickle.load(f)

""" sentences """
s = [] 

for t in t:

    print('Downloading and processing summary of "' + t + '" ...')

    """ list of summary word-tokenized sentences """
    ss = [nltk.word_tokenize(s) for s in z.tokenize(wi.summary(t))]
    
    s.extend(list(filter(c, ss)))

print("Number of selected sentences: {}".format(len(s)))

for s0 in [" ".join(l) for l in s]:
    print(s0)

print("".join(main.tag_viterbi(s,
    p['tagset'],
    p['known_words'],
    p['q_values'],
    p['e_values'])))

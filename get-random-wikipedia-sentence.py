import wikipedia as wi, nltk, main, pickle
import sys


# TODO: consider using https://en.wikipedia.org/wiki/Special:Random
# https://en.wikipedia.org/wiki/Special:RandomInCategory/Mathematics

# FIXME: this selects only uncategorized articles :<
" titles "
n = 5
if len(sys.argv) > 1:
    n = int(sys.argv[1])
t = wi.random(pages=n)
if type(t) is str: t = [t]

print('Random wikipedia articles: "' + '", "'.join(t) + '"')

z = nltk.data.load('tokenizers/punkt/english.pickle')

def c(s):
    return 10 < len(s) < 17

""" model parameters """
p = {}
for n in 'e_values known_words q_values tagset'.split():
    with open('parameters/' + n + '.pkl', 'rb') as f:
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

for tagg in [main.tag_viterbi(sen,
    p['tagset'],
    p['known_words'],
    p['q_values'],
    p['e_values']) for sen in s]:
    print(" ".join(map(lambda z: "/".join(map(str,z)),tagg)))

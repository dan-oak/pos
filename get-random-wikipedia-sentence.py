import wikipedia, nltk, main, pickle

titles = wikipedia.random(pages=1)
if type(titles) != list:
    titles = [titles]
print(titles)
sentences = []

def select_sentence(sentences):
    pass

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

object_names = 'e_values known_words q_values taglist'.split()
objects = {}
for name in object_names:
    with open('objects/' + name + '.pkl', 'rb') as object_file:
        objects[name] = pickle.load(object_file)

for title in titles:
    sentences = tokenizer.tokenize(wikipedia.summary(title))
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    print(main.viterbi(sentences,
        objects['taglist'],
        objects['known_words'],
        objects['q_values'],
        objects['e_values']))
    #selected sentence = select_sentence(wikipedia.summary(title))
    #sentences.append(selected_sentence)

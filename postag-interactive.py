import pickle
import main
import nltk

names = 'tagset known_words q_values e_values'.split()
objects = {}
for name in names:
    with open('parameters/' + name + '.pkl', 'rb') as object_file:
        objects[name] = pickle.load(object_file)

prompt = 'Sentence > '

input_string = None
while not input_string in ['q','quit','exit']:
    if not input_string:
        input_string = 'Enter an English sentence to tag its tokens with the respective parts of speech, -- this is an example.'
        print(prompt + input_string)
    else:
        input_string = input(prompt)
    sentence = nltk.word_tokenize(input_string)
    tagged = main.tag_viterbi(sentence, objects['tagset'], objects['known_words'], objects['q_values'], objects['e_values'])
    print('Tagged : ' + " ".join(["{0}/{1}".format(*x) for x in tagged]))

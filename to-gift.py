import main
import pickle
import sys, os
import nltk
import math

t,k,e,q = main.load_viterbi_parameters()
tl = list(t.difference({main.STAR,main.STOP}))

with open(sys.argv[1], 'r') as input_file:
    sentences = input_file.readlines()

output_dir = sys.argv[2]
filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]

number_width = int(math.log(len(sentences), 10))

for i, sentence in enumerate(sentences):
    tagged = main.tag_viterbi(nltk.word_tokenize(sentence), t,k,e,q)
    with open(os.path.join(output_dir, filename +
        str(i).zfill(number_width) + '.gift'), "w") as outf:
        outf.write("1. Визначити частини мови елементів речення.\n" +
            sentence)
        for tt in tagged:
            outf.write(tt[0] + ' {1:MULTICHOICE:' +
                "~".join("=" + t if t == tt[1] else t for t in tl) + 
                '}\n')

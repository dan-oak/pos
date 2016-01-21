import sys, os.path

def main():
    if len(sys.argv) < 3:
        print("Usage: python " + os.path.basename(__file__) +
                " <tagger output> <reference file>")
        exit(1)

    with open(sys.argv[1], "r") as f:
        user_sentences = f.readlines()

    with open(sys.argv[2], "r") as f:
        correct_sentences = f.readlines()

    num_correct = 0
    total = 0

    for user_sent, correct_sent in zip(user_sentences, correct_sentences):
        user_tok = user_sent.split()
        correct_tok = correct_sent.split()

        # TODO: compare corresponding sentences
        if len(user_tok) != len(correct_tok):
            continue

        for u, c in zip(user_tok, correct_tok):
            if u == c:
                num_correct += 1
            total += 1

    score = float(num_correct) / total * 100

    print("{:d}% of tags are correct".format(int(score)))


if __name__ == "__main__": main()

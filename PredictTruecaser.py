from Truecaser import *
import os
import cPickle
import nltk
import string
import argparse
import fileinput

    
if __name__ == "__main__":       
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to truecase, if empty, STDIN is used')
    parser.add_argument('-d', '--distribution_object', help='language distribution file', type=os.path.abspath, required=True)
    args = parser.parse_args()

    f = open(args.distribution_object, 'rb')
    uniDist = cPickle.load(f)
    backwardBiDist = cPickle.load(f)
    forwardBiDist = cPickle.load(f)
    trigramDist = cPickle.load(f)
    wordCasingLookup = cPickle.load(f)
    f.close()
    
    for sentence in fileinput.input(files=args.files):
        tokensCorrect = nltk.word_tokenize(sentence)
        tokens = [token.lower() for token in tokensCorrect]
        tokensTrueCase = getTrueCase(tokens, 'title', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
        print(" ".join(tokensTrueCase))


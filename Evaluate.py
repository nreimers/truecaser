from __future__ import print_function
import os
import nltk
import sys
import random

if (sys.version_info > (3, 0)):
    import pickle as pkl
else: 
    #Python 2.7 imports
    import cPickle as pkl
    from io import open

from truecaser.AbstractTruecaser import AbstractTruecaser
from truecaser.GreedyTruecaser import GreedyTruecaser
from truecaser.StatisticalTruecaser import StatisticalTruecaser
from truecaser.NeuralTruecaser import NeuralTruecaser


#Train, Development and Test files from wikipedia for some languages
language = 'tr'
trainFile = 'wikipedia/%s/train_100k.txt' % language
devFile = 'wikipedia/%s/dev.txt' % language
testFile = 'wikipedia/%s/test.txt' % language


# :: Selection of the model ::

#GreedyTruecaser: Generates a dictionary of uni-, bi- and trigrams and sets the casing to the most likely one
#model = GreedyTruecaser()
#model.use_bigrams = False
#model.use_trigrams = False


#StatisticalTruecaser: Uses a statistical model to infer the best casing. Statistical model is based on uni-, bi- and trigram statistics
model = StatisticalTruecaser()

#NeuralTruecaser: Uses a deep neural network to truecase sentences. Note: Neural networks are slow to train and to evaluate!
#model = NeuralTruecaser()
#model.set_development_set([line for line in open(devFile)])


if language == 'tr':
    model.title_case_unknown_tokens = False

#:: Output path for storing  models ::
modelpath = "models/%s_%s_%d.obj" % (language, model.__class__.__name__, random.randint(0,10000000))


# ------------------------------------------------------------------------ #

def read_sentences(filepath):
    sentencens = []
    with open(filepath, encoding='utf-8') as fIn:
        for line in fIn:
            sentencens.append(line.strip())
    return sentencens


def evaluate(model, test_sentences, print_errors = False):  
    test_sentences = map(nltk.word_tokenize, test_sentences)
        
    correctTokens = 0
    totalTokens = 0
    
    numUppercaseGold = 0
    numUppercasePred = 0
    numUppercaseCorrect = 0
    
    diff_length_error_count = 0
    
    for sentence in test_sentences:        
        tokens_lowercased = [token.lower() for token in sentence]
        tokens_truecased = model.truecase(tokens_lowercased, True, True)
        
        if len(tokens_lowercased) != len(tokens_truecased):
            diff_length_error_count += 1         
            continue
            
        assert(len(tokens_lowercased) == len(tokens_truecased))
        perfectMatch = True
        for idx in range(len(sentence)):
            totalTokens += 1
            if sentence[idx] == tokens_truecased[idx]:
                correctTokens += 1
            else:
                perfectMatch = False
                
            # A not lower cased word
            if sentence[idx] != sentence[idx].lower():
                numUppercaseGold += 1
                
                if sentence[idx] == tokens_truecased[idx]:
                    numUppercaseCorrect += 1
                    
            if tokens_truecased[idx] != tokens_truecased[idx].lower():
                numUppercasePred += 1
                
                
        if print_errors and not perfectMatch:
            print("Correct:  ", sentence)
            print("TrueCased:", tokens_truecased)            
            print("-------------------")
                
    accuracy = (correctTokens / float(totalTokens))
    precision = numUppercaseCorrect / float(numUppercasePred)
    recall = numUppercaseCorrect / float(numUppercaseGold)
    f1 = 2*precision*recall / (recall + precision)
    
    print("Accuracy: %.2f%%" % (accuracy*100))
    print("Precision: %.2f%%" % (precision*100))
    print("Recall: %.2f%%" % (recall*100))
    print("F1: %.2f%%" % (f1*100))  
    print("Truecased sentence with different length: %d" % diff_length_error_count )
   
# ------------------------------------------------------------------------ #

if __name__ == '__main__':      
    train = True
    
    if os.path.exists(modelpath):
        print("Load model: %s" % modelpath)
        model = AbstractTruecaser.load(modelpath)
        train = False #Do not train if we load a model
    
    print("Model-type: ", model.__class__)
    trainSentences = read_sentences(trainFile)
    testSentences = read_sentences(testFile)
    
    if train:
        print("Train on "+trainFile)
        model.train(trainSentences)
    
       
    print("Evaluate on "+testFile)
    evaluate(model, testSentences)
    
 
    if train: #Store model when it was trained
        print("Store model in: %s" % modelpath)
        model.save(modelpath)
        
    print("--DONE--")

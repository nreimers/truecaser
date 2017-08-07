import sys
import nltk
import string
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: 
    #Python 2.7 imports
    import cPickle as pkl
    from io import open

class AbstractTruecaser:
    def train(self, sentences, input_tokenized = False):
        raise NotImplementedError("Please implement this method")
    
    def truecase(self, sentence, input_tokenized=False, output_tokenized=False, title_case_start_sentence=True):
        raise NotImplementedError("Please implement this method")
    
    def save(self, modelpath):  
        #raise NotImplementedError("Please implement this method")
        with open(modelpath, 'wb') as fOut:        
            pkl.dump(self, fOut)
    
    
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)
    
    def untokenize(self, tokens):
        return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
    
    @staticmethod
    def load(modelpath):         
        with open(modelpath, 'rb') as f:
            model = pkl.load(f)
        
        return model
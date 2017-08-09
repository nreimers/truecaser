from __future__ import print_function
import nltk
from AbstractTruecaser import AbstractTruecaser
import pickle as pkl 
import string

class GreedyTruecaser(AbstractTruecaser):
    unigrams = {}
    bigrams = {}
    trigrams = {}
    stopwords = {}
    
    use_bigrams = True
    use_trigrams = True
    
    uni_freq_threshold = 0
    ngram_freq_threshold = 10
    
    title_case_unknown_tokens = True
    print_errors = False
    
     
    
    def train(self, sentences, input_tokenized = False):
        if not input_tokenized:
            sentences = map(self.tokenize, sentences)
        
      
        
        unigrams_stats = dict()
        
        token_stats = nltk.FreqDist()
        
        #Compute freq. of casing for tokens
        for sentence in sentences:
            if not self.check_sentence_sanity(sentence):
                continue
            
            for tokenIdx in range(1, len(sentence)):
                token = sentence[tokenIdx]
                token_norm = self.normalize_token(token)
                
                token_stats[token_norm] += 1
                
                if token_norm not in unigrams_stats:
                    unigrams_stats[token_norm] = nltk.FreqDist()
                
                unigrams_stats[token_norm][token] += 1
                
                        
        #Set stopwords to the most commont words    
        for token, freq in token_stats.most_common(100):
            self.stopwords[token] = True
            
        #Prune unigrams, keep only most frequent casings
        unigrams = dict()
        for token_norm in unigrams_stats:
            most_common_casing, freq = unigrams_stats[token_norm].most_common(1)[0]
            if freq < self.uni_freq_threshold:
                continue
            unigrams[token_norm] = most_common_casing
            
        self.unigrams = unigrams
        
        #Cleanup
        del unigrams_stats
        del token_stats    
            
            
        # :: Bigrams ::
        if self.use_bigrams:
            self.bigrams = self._ngram_stats(sentences, 2)
            
        # :: Trigrams ::
        if self.use_trigrams:
            self.trigrams = self._ngram_stats(sentences, 3)
            
            
    
                  
    def truecase(self, sentence, input_tokenized=False, output_tokenized=False, title_case_start_sentence=True):            
        if not input_tokenized:
            sentence = self.tokenize(sentence)
            
        sentence_norm = map(self.normalize_token, sentence)
        truecased = [None] * len(sentence_norm)
        
        
        # Test for trigrams
        tokenIdx = 0
        while self.use_trigrams and tokenIdx < len(sentence)-2:   
            trigram = " ".join(sentence_norm[tokenIdx:tokenIdx+3])
            
            if trigram in self.trigrams:
                phrase = self.trigrams[trigram]
                
                for pos in range(len(phrase)):    
                    if truecased[tokenIdx+pos] == None:       
                        truecased[tokenIdx+pos] = phrase[pos]                   
            tokenIdx += 1
                
        # Test for bigrams
        tokenIdx = 0
        while self.use_bigrams and tokenIdx < len(sentence)-1:             
            bigram = " ".join(sentence_norm[tokenIdx:tokenIdx+2])
            
            if bigram in self.bigrams:
                phrase = self.bigrams[bigram]     
                for pos in range(len(phrase)):    
                    if truecased[tokenIdx+pos] == None:       
                        truecased[tokenIdx+pos] = phrase[pos]                
            
            tokenIdx += 1
            
        # Fall back to unigrams
        tokenIdx = 0
        while tokenIdx < len(sentence):             
            if truecased[tokenIdx] != None:
                tokenIdx += 1
                continue
                   
            token_norm = sentence_norm[tokenIdx] 
            token_cased = token_norm
            
            if token_norm in self.unigrams:
                token_cased = self.unigrams[token_norm]
            elif self.title_case_unknown_tokens:
                token_cased = token_cased.title()
            
                        
            truecased[tokenIdx] = token_cased
            tokenIdx += 1
            
        
        if title_case_start_sentence:
            #Title case the first token in a sentence
            truecased[0] = truecased[0].title() if truecased[0].islower() else truecased[0]
            
            if truecased[0] == '"':
                truecased[1] = truecased[1].title() if truecased[1].islower() else truecased[1]
        
        if not output_tokenized:
            truecased = self.untokenize(truecased)
            
        return truecased
    


    
    def _ngram_stats(self, sentences, size):
        ngrams_stats = dict()
        
        # Compute casing statistics for ngrams
        for sentence in sentences:
            if not self.check_sentence_sanity(sentence):
                continue
            
            for ngram in self._ngrams(sentence, size, 2):
                ngram_norm = " ".join(map(self.normalize_token, ngram))
                ngram_str = " ".join(ngram)
                
                if ngram_norm not in ngrams_stats:
                    ngrams_stats[ngram_norm] = nltk.FreqDist()
                
                ngrams_stats[ngram_norm][ngram_str] += 1
            
        #Prune ngrams, keep only most frequent casings
        ngrams = {}
        for token_norm in ngrams_stats:            
            most_common_casing, freq = ngrams_stats[token_norm].most_common(1)[0]
            
            if freq < self.ngram_freq_threshold:
                continue
            
            words = most_common_casing.split() 
            
            # Skip ngrams that have punctuation in them
            contains_punctuation= False
            for word in words:
                if word in [',', '.', '"', "'", '``', '(', ')', '{', '}', '[', ']', "''"]:
                    contains_punctuation = True
                    break
            
            if contains_punctuation:
                continue
            
            # Skip ngrams that start with a stopword
            if words[0] in self.stopwords:
                continue
            
            #Would we produce the same casing with unigrams?                     
            words_truecased = self.truecase(words, True, True, title_case_start_sentence=False)
            
            redundantInfo = True            
            for wordIdx in range(len(words)):
                if words[wordIdx] != words_truecased[wordIdx]:
                    redundantInfo = False
                    break
            
            if redundantInfo:
                continue
            
            
            ngrams[token_norm] = words
        
        return ngrams
    
    def check_sentence_sanity(self, sentence):
        """ Checks the sanity of the sentence. Reject too short sentences"""
        return len(sentence) >= 6 and not " ".join(sentence).isupper()
          
    def normalize_token(self, word):
        return word.lower()
    
    
    def _ngrams(self, sentence, size, offset=0):
        for pos in range(offset, len(sentence)-(size-1)):
            yield sentence[pos:pos+size]
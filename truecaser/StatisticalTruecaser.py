from __future__ import print_function
import nltk
import math
import string
from AbstractTruecaser import AbstractTruecaser

class StatisticalTruecaser(AbstractTruecaser):
    
    
    def __init__(self):
        self.uniDist = nltk.FreqDist()
        self.backwardBiDist = nltk.FreqDist() 
        self.forwardBiDist = nltk.FreqDist() 
        self.trigramDist = nltk.FreqDist() 
        self.wordCasingLookup = {}
        self.title_case_unknown_tokens = True

    
    def train(self, sentences, input_tokenized = False):
        if not input_tokenized:
            sentences = map(self.tokenize, sentences)
       
        
        cleanedSentences = []
        for sentence in sentences:
            if self._check_sentence_sanity(sentence):
                cleanedSentences.append(sentence)
        
        
        print("Create unigram lookup")
        # :: Create unigram lookup ::
        for sentence in cleanedSentences:        
            for tokenIdx in xrange(1, len(sentence)):
                word = sentence[tokenIdx]
                self.uniDist[word] += 1
                            
                if word.lower() not in self.wordCasingLookup:
                    self.wordCasingLookup[word.lower()] = set()
                
                self.wordCasingLookup[word.lower()].add(word)
                
        print("Create bi+trigram lookup")
        # :: Create backward + forward bigram lookup + trigram lookup ::
        for sentence in cleanedSentences:        
            # :: Create bigram lookuo ::
            for tokenIdx in range(2, len(sentence)): #Start at 2 to skip first word in sentence
                word = sentence[tokenIdx]
                wordLower = word.lower()
                
                if wordLower in self.wordCasingLookup and len(self.wordCasingLookup[wordLower]) >= 2: #Only if there are multiple options
                    prevWord = sentence[tokenIdx-1]
                    
                    self.backwardBiDist[prevWord+"_"+word] +=1
                    
                    if tokenIdx < len(sentence)-1:
                        nextWord = sentence[tokenIdx+1].lower()
                        self.forwardBiDist[word+"_"+nextWord] += 1
                        
            # :: Create trigram lookup ::        
            for tokenIdx in range(2, len(sentence)-1): #Start at 2 to skip first word in sentence
                prevWord = sentence[tokenIdx-1]
                curWord = sentence[tokenIdx]
                curWordLower = word.lower()
                nextWordLower = sentence[tokenIdx+1].lower()
                
                if curWordLower in self.wordCasingLookup and len(self.wordCasingLookup[curWordLower]) >= 2: #Only if there are multiple options   
                    self.trigramDist[prevWord+"_"+curWord+"_"+nextWordLower] += 1
        

    
    def train_from_ngram_file(self, bigramFile, trigramFile):
        """
        Updates the FrequencyDistribitions based on an ngram file,
        e.g. the ngram file of http://www.ngrams.info/download_coca.asp
        """
        for line in open(bigramFile):
            splits = line.strip().split('\t')
            cnt, word1, word2 = splits
            cnt = int(cnt)
            
            # Unigram
            if word1.lower() not in self.wordCasingLookup:
                self.wordCasingLookup[word1.lower()] = set()
                
            self.wordCasingLookup[word1.lower()].add(word1)
            
            if word2.lower() not in self.wordCasingLookup:
                self.wordCasingLookup[word2.lower()] = set()
                
            self.wordCasingLookup[word2.lower()].add(word2)
            
            
            self.uniDist[word1] += cnt
            self.uniDist[word2] += cnt
            
            # Bigrams
            self.backwardBiDist[word1+"_"+word2] +=cnt
            self.forwardBiDist[word1+"_"+word2.lower()] += cnt
            
            
        #Tigrams
        for line in open(trigramFile):
            splits = line.strip().split('\t')
            cnt, word1, word2, word3 = splits
            cnt = int(cnt)
            
            self.trigramDist[word1+"_"+word2+"_"+word3.lower()] += cnt
            
    
    def truecase(self, sentence, input_tokenized=False, output_tokenized=False, title_case_start_sentence=True):     
        """
        Returns the true case for the passed tokens.
        @param tokens: Tokens in a single sentence
      
        """
        if not input_tokenized:
            sentence = self.tokenize(sentence)
            
        tokensTrueCase = []
        for tokenIdx in range(len(sentence)):
            token = sentence[tokenIdx]
            if token in string.punctuation or token.isdigit():
                tokensTrueCase.append(token)
            else:
                if token in self.wordCasingLookup:
                    if len(self.wordCasingLookup[token]) == 1:
                        tokensTrueCase.append(list(self.wordCasingLookup[token])[0])
                    else:
                        prevToken = tokensTrueCase[tokenIdx-1] if tokenIdx > 0  else None
                        nextToken = sentence[tokenIdx+1] if tokenIdx < len(sentence)-1 else None
                        
                        bestToken = None
                        highestScore = float("-inf")
                        
                        for possibleToken in self.wordCasingLookup[token]:
                            score = self._score(prevToken, possibleToken, nextToken)
                               
                            if score > highestScore:
                                bestToken = possibleToken
                                highestScore = score
                            
                        tokensTrueCase.append(bestToken)
                        
                   
                        
                else: #Token out of vocabulary
                    if self.title_case_unknown_tokens:
                        tokensTrueCase.append(token.title())
                    else:
                        tokensTrueCase.append(token.lower())
                    
                        
        if title_case_start_sentence:
            #Title case the first token in a sentence
            tokensTrueCase[0] = tokensTrueCase[0].title() if tokensTrueCase[0].islower() else tokensTrueCase[0]
            
            if tokensTrueCase[0] == '"':
                tokensTrueCase[1] = tokensTrueCase[1].title() if tokensTrueCase[1].islower() else tokensTrueCase[1]
                
        if not output_tokenized:
            tokensTrueCase = self.untokenize(tokensTrueCase)
            
        return tokensTrueCase

    
    def _check_sentence_sanity(self, sentence):
        """ Checks the sanity of the sentence. Reject too short sentences"""
        return len(sentence) >= 6 and not " ".join(sentence).isupper()
    
    def _score(self, prevToken, possibleToken, nextToken):
        pseudoCount = 5.0
        
        #Get Unigram Score
        nominator = self.uniDist[possibleToken]+pseudoCount    
        denominator = 0    
        for alternativeToken in self.wordCasingLookup[possibleToken.lower()]:
            denominator += self.uniDist[alternativeToken]+pseudoCount
            
        unigramScore = nominator / denominator
            
            
        #Get Backward Score  
        bigramBackwardScore = 1
        if prevToken != None and self.backwardBiDist != None:  
            nominator = self.backwardBiDist[prevToken+'_'+possibleToken]+pseudoCount
            denominator = 0    
            for alternativeToken in self.wordCasingLookup[possibleToken.lower()]:
                denominator += self.backwardBiDist[prevToken+'_'+alternativeToken]+pseudoCount
                
            bigramBackwardScore = nominator / denominator
            
        #Get Forward Score  
        bigramForwardScore = 1
        if nextToken != None and self.forwardBiDist != None:  
            nextToken = nextToken.lower() #Ensure it is lower case
            nominator = self.forwardBiDist[possibleToken+"_"+nextToken]+pseudoCount
            denominator = 0    
            for alternativeToken in self.wordCasingLookup[possibleToken.lower()]:
                denominator += self.forwardBiDist[alternativeToken+"_"+nextToken]+pseudoCount
                
            bigramForwardScore = nominator / denominator
            
            
        #Get Trigram Score  
        trigramScore = 1
        if prevToken != None and nextToken != None and self.trigramDist != None:  
            nextToken = nextToken.lower() #Ensure it is lower case
            nominator = self.trigramDist[prevToken+"_"+possibleToken+"_"+nextToken]+pseudoCount
            denominator = 0    
            for alternativeToken in self.wordCasingLookup[possibleToken.lower()]:
                denominator += self.trigramDist[prevToken+"_"+alternativeToken+"_"+nextToken]+pseudoCount
                
            trigramScore = nominator / denominator
            
        result = math.log(unigramScore) + math.log(bigramBackwardScore) + math.log(bigramForwardScore) + math.log(trigramScore)
        #print "Scores: %f %f %f %f = %f" % (unigramScore, bigramBackwardScore, bigramForwardScore, trigramScore, math.exp(result))
      
      
        return result
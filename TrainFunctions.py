import nltk

def getCasing(word):  
    """ Returns the casing of a word"""
    if len(word) == 0:
        return 'other'
    elif word.isdigit(): #Is a digit
        return 'numeric'
    elif word.islower(): #All lower case
        return 'allLower'
    elif word.isupper(): #All upper case
        return 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        return 'initialUpper'
    
    return 'other'


def checkSentenceSanity(sentence):
    """ Checks the sanity of the sentence. If the sentence is for example all uppercase, it is recjected"""
    caseDist = nltk.FreqDist()
    
    for token in sentence:
        caseDist[getCasing(token)] += 1
    
    if caseDist.most_common(1)[0][0] != 'allLower':        
        return False
    
    return True

def updateDistributionsFromSentences(text, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    """
    Updates the NLTK Frequency Distributions based on a list of sentences.
    text: Array of sentences.
    Each sentence must be an array of Tokens.
    """
    # :: Create unigram lookup ::
    for sentence in text:
        if not checkSentenceSanity(sentence):
            continue
        
        for tokenIdx in xrange(1, len(sentence)):
            word = sentence[tokenIdx]
            uniDist[word] += 1
                        
            if word.lower() not in wordCasingLookup:
                wordCasingLookup[word.lower()] = set()
            
            wordCasingLookup[word.lower()].add(word)
            
    
    # :: Create backward + forward bigram lookup ::
    for sentence in text:
        if not checkSentenceSanity(sentence):
            continue
        
        for tokenIdx in xrange(2, len(sentence)): #Start at 2 to skip first word in sentence
            word = sentence[tokenIdx]
            wordLower = word.lower()
            
            if wordLower in wordCasingLookup and len(wordCasingLookup[wordLower]) >= 2: #Only if there are multiple options
                prevWord = sentence[tokenIdx-1]
                
                backwardBiDist[prevWord+"_"+word] +=1
                
                if tokenIdx < len(sentence)-1:
                    nextWord = sentence[tokenIdx+1].lower()
                    forwardBiDist[word+"_"+nextWord] += 1
                    
    # :: Create trigram lookup ::
    for sentence in text:
        if not checkSentenceSanity(sentence):
            continue
        
        for tokenIdx in xrange(2, len(sentence)-1): #Start at 2 to skip first word in sentence
            prevWord = sentence[tokenIdx-1]
            curWord = sentence[tokenIdx]
            curWordLower = curWord.lower()
            nextWordLower = sentence[tokenIdx+1].lower()
            
            if curWordLower in wordCasingLookup and len(wordCasingLookup[curWordLower]) >= 2: #Only if there are multiple options   
                trigramDist[prevWord+"_"+curWord+"_"+nextWordLower] += 1
    
            
 

def updateDistributionsFromNgrams(bigramFile, trigramFile, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    """
    Updates the FrequencyDistribitions based on an ngram file,
    e.g. the ngram file of http://www.ngrams.info/download_coca.asp
    """
    for line in open(bigramFile):
        splits = line.strip().split('\t')
        cnt, word1, word2 = splits
        cnt = int(cnt)
        
        # Unigram
        if word1.lower() not in wordCasingLookup:
            wordCasingLookup[word1.lower()] = set()
            
        wordCasingLookup[word1.lower()].add(word1)
        
        if word2.lower() not in wordCasingLookup:
            wordCasingLookup[word2.lower()] = set()
            
        wordCasingLookup[word2.lower()].add(word2)
        
        
        uniDist[word1] += cnt
        uniDist[word2] += cnt
        
        # Bigrams
        backwardBiDist[word1+"_"+word2] +=cnt
        forwardBiDist[word1+"_"+word2.lower()] += cnt
        
        
    #Tigrams
    for line in open(trigramFile):
        splits = line.strip().split('\t')
        cnt, word1, word2, word3 = splits
        cnt = int(cnt)
        
        trigramDist[word1+"_"+word2+"_"+word3.lower()] += cnt
        
        

        

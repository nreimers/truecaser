# -*- coding: utf-8 -*-
"""
Run by:
python lstm.py [LangCode] [ExpName]  [LSTMSizes]
"""
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import time
import sys
import nltk
import math
import random 
import tempfile
from truecaser.AbstractTruecaser import AbstractTruecaser

class NeuralTruecaser(AbstractTruecaser):
    def __init__(self):
        self.lstmSizes = [100]
        self.dev_sentences = []
        
    def set_development_set(self, sentences):
        self.dev_sentences = sentences
        
    def train(self, sentences, input_tokenized = False, max_epochs = 10):
        if input_tokenized == True:
            sentences = map(self.untokenize, sentences)
          
        alphabet = self._mostCommonChars(sentences, 100)
        
        alphabetOutput = ''.join(sorted(alphabet))
        alphabetOutput = alphabetOutput.encode("utf8")
        print("Alphabet: ", alphabetOutput)
        
        self.vocab = {alphabet[idx]: idx for idx in xrange(len(alphabet))}
        self.vocab['UNKNOWN_CHAR'] = len(self.vocab) #Add unknown char
        vocabSize = len(self.vocab)
        charDim = 25
    
        
        model = Sequential()
        model.add(Embedding(input_dim=vocabSize, output_dim=charDim, trainable=True))
        
        for lstmSize in self.lstmSizes:
            model.add(Bidirectional(LSTM(lstmSize, return_sequences=True, dropout_W=0.25, dropout_U=0.25)))
        
        model.add(TimeDistributed(Dense(1, activation='sigmoid'), name='sigmoid_output'))            
                
        opt = Nadam(clipnorm=1)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        
        model.summary()
        self.model = model
        
        print("Read in trainings data")
        trainData = []
        for sentence in sentences:
            x = self._getChars(sentence)
            y = self._getCasing(sentence)
            trainData.append((x,y))  
            
        print("Start training the network for %d epochs" % max_epochs)     
        
        last_dev_acc = 0
        for epoch in range(max_epochs):
            print("\nEpoch %d" % (epoch+1))   
            self._train_network(trainData)      
            dev_acc = self.evaluate_on_development()
            
            if dev_acc < last_dev_acc:
                print("Early stopping, accuracy on development set decreased")
            
    def truecase(self, sentence, input_tokenized=False, output_tokenized=False, title_case_start_sentence=True):  
        if input_tokenized:
            sentence = self.untokenize(sentence)
            
        sentence = sentence.lower()
        
        x = self._getChars(sentence)
        x = np.asarray([x])
        
        predictions = self.model.predict_on_batch(x)
        linePredictions = predictions[0]
       
        trueCasedLine = []
        for idx in xrange(len(linePredictions)):
            if linePredictions[idx] >= 0.5:
                trueCasedLine.append(sentence[idx].upper())
            else:
                trueCasedLine.append(sentence[idx])
               
        trueCasedLine = "".join(trueCasedLine)
         
        if output_tokenized:
            trueCasedLine = self.tokenize(trueCasedLine)
                
        return trueCasedLine
            
    def evaluate_on_development(self):
        start_time = time.time() 
        evalLines = self.dev_sentences   
        evalLines.sort(key=lambda x:len(x)) #Sort by sentence length
        
        evalData = []
        for line in evalLines:
            x = self._getChars(line)
            evalData.append((x,))
            
        evalRanges = self._getSentenceLengthRanges(evalData, sortData=False)
        
        numSent = 0
        numCorrectTokens = 0
        numTotalTokens = 0
        
        for dataRange in evalRanges:
            tokens = np.asarray([evalData[idx][0] for idx in xrange(dataRange[0], dataRange[1])])
            predictions = self.model.predict_on_batch(tokens)
            
    
    
            for predIdx in xrange(len(predictions)):
                numSent += 1
                line = evalLines[dataRange[0]+predIdx]
                lowerCaseLine = line.lower()
                trueCasedLine = ""
                linePredictions = predictions[predIdx]
                for idx in xrange(len(linePredictions)):
                    if linePredictions[idx] >= 0.5:
                        trueCasedLine += lowerCaseLine[idx].upper()
                    else:
                        trueCasedLine += lowerCaseLine[idx]
                
               
             
                correctTokens = line.split(" ")
                casedTokens = trueCasedLine.split(" ")
                
                for idx in xrange(len(correctTokens)):
                    numTotalTokens += 1
                    
                    if correctTokens[idx].lower() != casedTokens[idx].lower():
                        print("ERROR: correctTokens != casedTokens")
                    
                    if correctTokens[idx] == casedTokens[idx]:
                        numCorrectTokens += 1     
                        
                      
                
       
        accuracy = numCorrectTokens/float(numTotalTokens)
        
            
      
        print("Development-Accuracy: %.2f%%" % (accuracy*100))
     
        sys.stdout.flush()
        
        return accuracy
    
    def __getstate__(self):
 
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self.model, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'vocab': self.vocab, 'model_str': model_str }
        return d
    
    def __setstate__(self, state):         
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            self.model = keras.models.load_model(fd.name)
        self.vocab = state['vocab']
    
    def _train_network(self, trainData):
        trainRanges = self._getSentenceLengthRanges(trainData)
        miniBatchRanges = self._getMinibatchRanges(trainData, trainRanges)
        cnt = 0
        evalStep = 10000
       
        nextEval = evalStep
        
        
        for idx in xrange(len(miniBatchRanges)):  
            dataRange = miniBatchRanges[idx]
            tokens = np.asarray([trainData[idx][0] for idx in xrange(dataRange[0], dataRange[1])])
            labels = np.asarray([trainData[idx][1] for idx in xrange(dataRange[0], dataRange[1])])
            labels = np.expand_dims(labels, -1)
            self.model.train_on_batch(tokens, labels)
            
            cnt += dataRange[1]-dataRange[0]
            
            if cnt > nextEval:
                print("Trained %d sentences" % cnt)
                self.evaluate_on_development()
                nextEval += evalStep
                print("")
    
    def _getChars(self, line):
        data = []
        vocab = self.vocab
        
        line = line.lower()
        for c in line:
            if c in vocab:
                data.append(vocab[c])
            else:
                data.append(vocab['UNKNOWN_CHAR'])
        
        return data


    def _getCasing(self, line):
        data = []    
        for c in line:
            data.append(1 if c.isupper() else 0)    
        
        return data
                
    def _mostCommonChars(self, sentences, num=100):
        fd = nltk.FreqDist()
        linesRead = 0
        for sentence in sentences: 
            for c in sentence:
                fd[c] += 1
            
            linesRead += 1
                
            if linesRead > 10000:
                break
        
        alphabet = ""
        for char, _ in fd.most_common(num):
            alphabet += char
        
        return alphabet
    
    def _getSentenceLengthRanges(self, data, sortData=True):    
        if sortData:
            data.sort(key=lambda x:len(x[0])) #Sort train matrix by sentence length
        
        dataRanges = []
        oldSentLength = len(data[0][0])            
        idxStart = 0
         
        #Find start and end of ranges with sentences with same length
        for idx in xrange(len(data)):     
            sentLength = len(data[idx][0])    
            if sentLength != oldSentLength:
                dataRanges.append((idxStart, idx))
                idxStart = idx
                
            
            oldSentLength = sentLength
            
        dataRanges.append((idxStart, len(data))) #Add last sentence   
        return dataRanges


 
    def _getMinibatchRanges(self, trainData, trainRanges, miniBatchSize=64):
        #Break up ranges into smaller mini batch sizes
        miniBatchRanges = []
        for batchRange in trainRanges:
            rangeLen = batchRange[1]-batchRange[0]        
            
            bins = int(math.ceil(rangeLen/float(miniBatchSize)))
            binSize = int(math.ceil(rangeLen / float(bins)))
            
            for binNr in xrange(bins):
                startIdx = binNr*binSize+batchRange[0]
                endIdx = min(batchRange[1],(binNr+1)*binSize+batchRange[0])
                miniBatchRanges.append((startIdx, endIdx))
        
        
        #1. Shuffle sentences that have the same length
        x = trainData
        for dataRange in trainRanges:
            for i in reversed(xrange(dataRange[0]+1, dataRange[1])):
                # pick an element in x[:i+1] with which to exchange x[i]
                j = random.randint(dataRange[0], i)
                x[i], x[j] = x[j], x[i]
           
        #2. Shuffle the order of the mini batch ranges       
        random.shuffle(miniBatchRanges)
        
        return miniBatchRanges   
    

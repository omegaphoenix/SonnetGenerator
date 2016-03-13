import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
import string

def countBigrams(words):
    bigram = {}
    prevWord = "\n"
    for word in words:
        if word != "\n":
            if prevWord != "\n":
                twoWords = prevWord + " " + word
                if twoWords in bigram:
                    bigram[twoWords] += 1
                else:
                    bigram[twoWords] = 1
            prevWord = word
    freq_bigram = []
    for word in bigram:
        if bigram[word] > 2:
            freq_bigram.append(word.split(" "))
    return freq_bigram

def writeToFile(outFile, words, bigram):
    outputFile = open(outFile, 'w')
    prev_word = ""
    for word in words:
        if prev_word != "\n":
            if [prev_word, word] in bigram:
                outputFile.write(prev_word + " " + word + "\n")
                prev_word = "" 
            else:
                if prev_word != "":
                    outputFile.write(prev_word + "\n")
                prev_word = word
        else:
            outputFile.write(prev_word)
            prev_word = word
    if prev_word != "\n":
        outputFile.write(prev_word + "\n")
    else:
        outputFile.write(prev_word)
    outputFile.close()

def readInFile(inFile):
    words = []
    for line in open(inFile, 'r'):
        if line == '\n':
            words.append(line)
        else:
            words.append(line.strip())
    return words

words = readInFile('complete_shakespeare_words.txt')
#words = readInFile('shakespeareWords.txt')
bigrams = countBigrams(words)
writeToFile('complete_bigrams.txt', words, bigrams)

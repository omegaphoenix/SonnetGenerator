import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
import string

# I deleted pandas and beautifulSoup because I didn't think we needed
# it just yet. I could be wrong though.

# Parsing a collection of sonnets into an aggregated array
# of words.
def parseStringToWords(trainingArray):
    aggregatedWords = []

    for sonnet in trainingArray:
        sonnetWords = sonnet.split()
        aggregatedWords.extend(sonnetWords)
        aggregatedWords.append("\n")

    return aggregatedWords

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

# Taking an array of words, and outputting it into a designated text file.
def writeToFile(outFile, words):
    outputFile = open(outFile, 'w')

    for word in words:
        if word != "\n":
            outputFile.write(word + "\n")
        else:
            outputFile.write(word)
    outputFile.close()

# Taking an array of words, and outputting it into a designated text file.
def writeToFile2(outFile, words, bigram):
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

# Reading in a file, and parsing it so that each sonnet occupies an
# index in the array.
def readInFile(inFile):
    input_file = open(inFile, 'r')

    # Each index contains a sonnet, with punctuation and capitalization stripped.
    train = []

    # String buffer to capture the stuff.
    sonnet = ""

    num_lines = 0

    # Fill train with the training examples
    for line in input_file.readlines():

        # Between sonnets
        if line in ['\n', '\r\n']:

            # If there is the wrong number of lines, skip
            if num_lines == 14:

                # makeTrans will just remove all punctuation from
                # the words. This solves the apostrophe problems.
                train.append(sonnet.translate(string.maketrans("",""),\
                                              string.punctuation))

            num_lines = 0
            sonnet = ""

        # Add line to sonnet
        else:

            # Ignoring lines that are numbers.
            if len(line.strip().split()) != 1:

                # Since maketrans removes all punctuation
                # we will have two lines connected. So,
                # each line will end with a space to make sure that
                # doesn't happen.
                sonnet += line.lower().strip() + " "
                num_lines += 1

    input_file.close()
    return train

# Running the script
sonnets = readInFile('shakespeare.txt')
words = parseStringToWords(sonnets)
bigrams = countBigrams(words)
writeToFile2('shakespeareWordsBigrams.txt', words, bigrams)

spenserSonnets = readInFile('spenser.txt')
spenserWords = parseStringToWords(spenserSonnets)
writeToFile('spenserWords.txt', spenserWords)

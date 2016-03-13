# This file is here to analyze what the states are?
# We can look via syllables and what not

import nltk
from nltk.corpus import cmudict

def getData(inFile):
    dataFile = open(inFile, 'r')

    totalData = []
    state = []
    for line in dataFile.readlines():
        if line != "\n":
            state.append(line.strip())

        else:
            totalData.append(state)
            state = []

    return totalData

def partsOfSpeech(inputStates):
    partsOfSpeech = {}
    states = []
    
    for state in inputStates:
        tokenizedWords = nltk.pos_tag(state)
        for tple in tokenizedWords:
            print tple
            if tple[1] not in partsOfSpeech:
                partsOfSpeech[tple[1]] = 1
            else:
                partsOfSpeech[tple[1]] += 1
        states.append(partsOfSpeech)
        partsOfSpeech = {}
    return states

def getSyllables(inputStates):
    d = cmudict.dict()
    states = []
    syllables = {}
    for state in inputStates:
        for word in state:
            try:
                numSyl = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]

                if numSyl[0] not in syllables:
                    syllables[numSyl[0]] = 1
                else:
                    syllables[numSyl[0]] += 1
            except KeyError:
                pass
        states.append(syllables)
        syllables = {}
    return states

def outputToFile(header, array):
    outFile = open(header + ".txt", "w")

    for idx, d in enumerate(array):
        outFile.write("State " + str(idx + 1) + "\n")
        for key in d.keys():
            outFile.write("Key: " + str(key) + "\t Value:" + str(d[key]) + "\n")
        outFile.write("\n")
    outFile.close()

words = getData("topWords.txt")
statePoS = partsOfSpeech(words)
print statePoS
outputToFile("PoS", statePoS)

syllables = getSyllables(words)
print syllables
outputToFile("syllable", syllables)
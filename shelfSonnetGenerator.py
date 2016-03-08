import numpy as np 
from hmmlearn import hmm

# Gonna try to generate a random poem.
# Using only Shakespearean stuff at first.

np.random.seed(55)

def getData(inFile):
    dataFile = open(inFile, 'r')

    totalData = []
    sonnet = []

    # Any time there is a "\n", we then make a new list.
    for line in dataFile.readlines():
        if line != "\n":
            sonnet.append(line.strip())

        else:
            totalData.append(sonnet)
            sonnet = []

    return totalData

def trainingHMM(sonnets):
    concatSonnet = np.concatenate(sonnets)
    concatSonnet = concatSonnet.reshape(-1, 1)

    lengths = []

    for sonnet in sonnets:
        lengths.append(len(sonnet))

    hmmModel = hmm.MultinomialHMM(n_components = 10).fit(concatSonnet, lengths)

    return hmmModel

def generateMaps(sonnets):
    wordMap = {}
    intMap = {}
    counter = 0
    setOfWords = []

    for sonnet in sonnets:
        for word in sonnet:
            word = word.strip()
            if word not in setOfWords:
                wordMap[word] = counter
                intMap[counter] = word
                counter += 1
                setOfWords.append(word)

    return (wordMap, intMap)

def mapWordToInt(sonnets, wordMap):
    newDataSet = []

    for sonnet in sonnets:
        intSonnet = []
        for word in sonnet:
            intRepresentation = wordMap[word]
            intSonnet.append(intRepresentation)
        newDataSet.append(intSonnet)
    return newDataSet

def mapIntToWord(line, intMap):
    lineTransp = line.T.tolist()
    print lineTransp
    oneLine = []
    for word in lineTransp[0]:
        wordRepresentation = intMap[word]
        oneLine.append(wordRepresentation)
    return oneLine

totalData = getData("shakespeareWords.txt")
wordMap, intMap = generateMaps(totalData)
intData = mapWordToInt(totalData, wordMap)

hmmModel = trainingHMM(intData)
lineOne, states = hmmModel.sample(10)
oneLine = mapIntToWord(lineOne, intMap)
print oneLine

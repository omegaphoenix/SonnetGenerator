# Based on solution to HW4 provided by Fabian Boemer and Kevin Tang
import sys
import numpy as np
import random
# for frobenius norm
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from itertools import chain
from nltk.corpus import cmudict
import heapq
trainingWords = []
def main():
    trainingWords = getData("complete_shakespeare_words.txt")
    #trainingWords = getData("shakespeareWords.txt")
    wordMap, intMap, wordCount = generateMaps(trainingWords)

    trainingSequence = mapWordToInt(trainingWords, wordMap)
    numObs = len(wordMap)

    # A, O are randomly initialized based on the number of states
    # and observations.

    for H_STATES in range(5, 15):
        A, O = randomlyInitialize(H_STATES, numObs)
        pi = generateStartProb(H_STATES)

        # Now, going to try to run baum_welch
        trainedPi, trainedA, trainedO = baum_welch(trainingSequence, A, O, pi, 5000)

        # Save matrices to file
        writeHMM('test_complete{}.txt'.format(H_STATES), trainedA, trainedO, trainedPi)

        good_words = analyzeHiddenStates(O, wordMap, intMap, wordCount)
        print good_words

def generate():
    trainingWords = getData("complete_shakespeare_words.txt")
    #trainingWords = getData("shakespeareWords.txt")
    wordMap, intMap, wordCount = generateMaps(trainingWords)
    for H_STATES in range(5,15):
        np.random.seed(13)
        random.seed(13)
        A, O, pi = loadHMM('test_complete{}.txt'.format(H_STATES))
        poem = ""
        poem += generatePoem(A, O, pi, wordMap, intMap)
        output_file = 'poem_complete{}.txt'.format(H_STATES)
        with open(output_file, 'w') as f:
            f.write(poem)
        print poem
    
def generateWord(A, O, pi, wordMap, intMap, prevState):
    word = ""
    next_state = 0
    prob = random.random()
    # Get first word
    if prevState == -1:
        for i in xrange(pi.shape[0]):
            if prob > pi[i]:
                prob -= pi[i]
            else:
                next_state = i
                break
        prob = random.random()
        for i in xrange(O.shape[1]):
            if prob > O[next_state, i]:
                prob -= O[next_state, i]
            else:
                word = intMap[i]
                break
    else:
        for i in xrange(A.shape[0]):
            if prob > A[prevState, i]:
                prob -= A[prevState, i]
            else:
                next_state = i
                break
        prob = random.random()
        for i in xrange(O.shape[1]):
            if prob > O[next_state, i]:
                prob -= O[next_state, i]
            else:
                word = intMap[i]
                break
    return word, next_state

def generateLine(A, O, pi, wordMap, intMap, prevState):
    line = ""
    state = prevState
    while countSyllabels(line) != 10:
        if countSyllabels(line) < 10:
            word, state = generateWord(A, O, pi, wordMap, intMap, state)
            line += word
            line += " "
        else:
            line = ""
            state = prevState
    return line + "\n", state

def generateCouplet(A, O, pi, wordMap, intMap, prevState):
    lines = []
    lastWord = []
    lastWordRhymes = []
    i = 0
    couplet = ""
    state = prevState
    while True:
        newLine, nextState = generateLine(A, O, pi, wordMap, intMap, state)
        words = newLine.split(" ")
        lastWord.append(words[len(words)-2])
        lastWordRhymes.append(rhyme(lastWord[i], 1, wordMap))
        print newLine
        # Skip if no rhyming words
        if len(lastWordRhymes[i]) == 0:
            print "no rhyme"
            lastWord.remove(lastWord[i])
            lastWordRhymes.remove(lastWordRhymes[i])
        else:
            # Check if it matches any of the previous lines
            for j in xrange(i):
                if (state == prevState and nextState == lines[j][2]):
                    if (lastWord[i] != lastWord[j]) and (lastWord[i] in lastWordRhymes[j]):
                        couplet = newLine + lines[j][0]
                        return couplet, nextState
                if (lines[j][1] == prevState and state == lines[j][2]):
                    if (lastWord[i] != lastWord[j]) and (lastWord[i] in lastWordRhymes[j]):
                        couplet = lines[j][0] + newLine
                        return couplet, state

            lines.append([newLine, state, nextState])
            # Randomly try new starting states
            if random.random() > 0.5:
                state = prevState
            else:
                state = nextState
            print i
            i += 1

def generateQuatrain(A, O, pi, wordMap, intMap, prevState):
    lines = []
    lastWord = []
    lastWordRhymes = []
    i = 0
    quatrain = ""
    line1 = ""
    line3 = ""
    state = prevState
    done = False
    while not done:
        newLine, nextState = generateLine(A, O, pi, wordMap, intMap, state)
        words = newLine.split(" ")
        lastWord.append(words[len(words)-2])
        lastWordRhymes.append(rhyme(lastWord[i], 2, wordMap))
        print newLine
        # Skip if no rhyming words
        if len(lastWordRhymes[i]) == 0:
            print "no rhyme"
            lastWord.remove(lastWord[i])
            lastWordRhymes.remove(lastWordRhymes[i])
        else:
            # Check if it matches any of the previous lines
            for j in xrange(i):
                if (lastWord[i] != lastWord[j]) and (lastWord[i] in lastWordRhymes[j]):
                    line1 = lines[j]
                    line3 = [newLine, state, nextState]
                    done = True
                    break
                if (lastWord[i] != lastWord[j]) and (lastWord[i] in lastWordRhymes[j]):
                    line1 = lines[j]
                    line3 = [newLine, state, nextState]
                    done = True
                    break
            lines.append([newLine, state, nextState])
            # Randomly try new starting states
            if random.random() > 0.5:
                state = prevState
            else:
                state = nextState
            print i
            i += 1
    # Reset variables
    i = 0
    lines = []
    lastWord = []
    lastWordRhymes = []
    done = False
    prevState = line1[2]
    nextState2 = line3[1]
    prevState2 = line3[2]
    state = prevState

    while not done:
        newLine, nextState = generateLine(A, O, pi, wordMap, intMap, state)
        words = newLine.split(" ")
        lastWord.append(words[len(words)-2])
        lastWordRhymes.append(rhyme(lastWord[i], 1, wordMap))
        print newLine
        # Skip if no rhyming words
        if len(lastWordRhymes[i]) == 0:
            print "no rhyme"
            lastWord.remove(lastWord[i])
            lastWordRhymes.remove(lastWordRhymes[i])
        else:
            # Check if it matches any of the previous lines
            for j in xrange(i):
                if True or (state == prevState and nextState == nextState2) and (lines[j][1] == prevState2):
                    if (lastWord[i] != lastWord[j]) and (lastWord[i] in lastWordRhymes[j]):
                        return line1[0] + newLine + line3[0] + lines[j][0], lines[j][2]
                if True or (state == prevState2) and (lines[j][1] == prevState and lines[j][2] == nextState2):
                    if (lastWord[i] != lastWord[j]) and (lastWord[i] in lastWordRhymes[j]):
                        return line1[0] + lines[j][0] + line3[0] + newLine, nextState
            lines.append([newLine, state, nextState])
            # Randomly try new starting states
            if random.random() > 0.5:
                state = prevState
            else:
                state = prevState2
            print i
            i += 1


def generatePoem(A, O, pi, wordMap, intMap):
    poem = ""
    state = -1
    for i in xrange(3):
        line, state = generateQuatrain(A, O, pi, wordMap, intMap, state)
        poem += line
    line, state = generateCouplet(A, O, pi, wordMap, intMap, state)
    poem += line
    return poem

def countSyllabels(line):
    syl = 0
    if line == "":
        return syl
    d = cmudict.dict()
    for word in line.split(" "):
        try:
            syla = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
            syl += syla[0]
        except KeyError:
            pass
    return syl

def checkRyhme(line1, line2, wordMap):
    words1 = line1.split(" ")
    last_word1 = words1[len(words1)-2]
    words2 = line2.split(" ")
    last_word2 = words2[len(words2)-2]
    if last_word1 in rhyme(last_word2, 1, wordMap):
        return True
    else:
        return False

def rhyme(inp, level, wordMap):
    entries = cmudict.entries()
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    if inp in rhymes:
        rhymes.remove(inp)
    return set(rhymes)

def test():
    # Tests from http://people.eng.unimelb.edu.au/tcohn/comp90042/HMM.py
    A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])
    pi = np.array([0.5, 0.2, 0.3])
    O = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
    states = UP, DOWN, UNCHANGED = 0, 1, 2
    observations = [UP, UP, DOWN]
    
    print "Test viterbi"
    t1 = viterbi(len(states), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP], A, O)
    t1_ans = "00222200"
    print t1 == t1_ans

    print "Test forward 1"
    t2 = forward([UP, UP, DOWN], A, O, pi)[1]
    t2_ans = 0.054397999999999995
    print abs(t2-t2_ans) < 1e-6
    
    print "Test forward 2"
    t3 = forward([UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP], A, O, pi)[1]
    t3_ans = 0.00023980534876400081
    print abs(t3-t3_ans) < 1e-6
    
    print "Test backward 1"
    t4 = backward([UP, UP, DOWN], A, O, pi)[1]
    t4_ans = 0.054397999999999995
    print abs(t4-t4_ans) < 1e-6
    
    # The answers we are basing off of did not account for over/underflow
    # Therefore, they make poor tests.
    # I'm leaving them in the code base to check that our code runs, but 
    # not making any comparisons.
    print "Test baum_welch 1"
    pi2, A2, O2 = baum_welch([[UP, UP, DOWN]], A, O, pi, 10)
    print forward([UP, UP, DOWN], A2, O2, pi2)[1]
    # print "0.47208638604110348"
    print forward([UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP], A2, O2, pi2)[1]
    # print "0.0"
    
    print "Test baum_welch 2"
    pi3, A3, O3 = baum_welch([[UP, UP, DOWN], [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP]], A, O, pi, 10)
    print forward([UP, UP, DOWN], A3, O3, pi3)[1]
    # print "0.23645963152993088"
    print forward([UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP], A3, O3, pi3)[1]
    # print "0.004774564161046658"

def test2():
    print "Test simulation"
    pi = np.array([0.5, 0.5])
    A = np.array([[0.85, 0.15],
                      [0.12, 0.88]])
    O = np.array([[0.8, 0.1, 0.1],
                      [0.0, 0.0, 1]])
    observ,states = simulate(1000, A, O, pi)
    pi1 = np.array([0.5, 0.5])
    A1 = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    O1 = np.array([[0.3, 0.3, 0.4],
                  [0.2, 0.5, 0.3]])
    pi2,A2,O2  = baum_welch([observ], A1, O1, pi1, 100)
    print 'Actual probabilities\n',pi
    print 'Estimated initial probabilities\n',pi2

    print 'Actual state transition probabililities\n',A
    print 'Estimated state transition probabililities\n',A2

    print 'Actual observation probabililities\n',O
    print 'Estimated observation probabililities\n',O2


def test_file():
    print "Testing Load and Write"
    # A, O = loadHMM('sequenceprediction1.txt')
    A, O, pi = loadHMM('test.txt')
    writeHMM('test2.txt', A, O, pi)
    A2, O2, pi2 = loadHMM('test2.txt')
    print A == A2
    print O == O2
    print pi == pi2

def loadHMM(filename):
    """ Loads a HMM file. Returns in format A, O
    """
    A = []  # transition matrix
    O = []  # observation matrix
    pi = [] # start probabilities

    with open(filename, 'r') as f:
        num_states, num_obs = [int(x)
                               for x in f.readline().strip().split('\t')]
        for i in range(num_states):
            A.append([float(x) for x in f.readline().strip().split('\t')])
        for i in range(num_states):
            O.append([float(x) for x in f.readline().strip().split('\t')])
        for x in f.readline().strip().split('\t'):
            pi.append(x)
    # print num_states, num_obs
    A = np.array(A)
    O = np.array(O)
    pi = np.array(pi)
    return (A, O, pi)

def writeHMM(filename, A, O, pi):
    """ Writes a HMM file. Follows the same format as the loadHMM function. """
    num_states = A.shape[0]
    num_obs = O.shape[1]
    with open(filename, 'w') as f:
        f.write(str(num_states))
        f.write('\t')
        f.write(str(num_obs))
        f.write('\n\r')
        for i in range(num_states):
            for j in range(num_states):
                f.write(str(A[i,j]))
                f.write('\t')
            # f.write(str(A[i,num_states-1]))
            f.write('\n\r')
        for i in range(num_states):
            for j in range(num_obs):
                f.write(str(O[i,j]))
                f.write('\t')
            # f.write(str(O[i,num_obs-1]))
            f.write('\n\r')
        for i in range(num_states):
            f.write(str(pi[i]))
            f.write('\t')
        f.write('\n\r')


def randomlyInitialize(num_states, num_obs):
    A = np.zeros((num_states, num_states))
    O = np.zeros((num_states, num_obs))

    # Randomizing each row.
    for elem in np.nditer(A, op_flags=['readwrite']):
        elem[...] = random.uniform(0, 1)

    for elem in np.nditer(O, op_flags=['readwrite']):
        elem[...] = random.uniform(0, 1)

    # Normalizing each row.
    A = normalize(A, axis=1, norm='l1')
    O = normalize(O, axis=1, norm='l1')

    return (A, O)


def viterbi(states, obs, A, O):
    """ Finds the max-probability state sequence for a given HMM and observation
        using the Viterbi Algorithm. This is a dynamic programming approach.
        The function uses 'prob' and 'seq' to store the probability and the
        sequence, respectively, of the most-likely sequences at each length.
        Arguments: states the number of states
                   obs    an array of observations
                   A      the transition matrix
                   O      the observation matrix
        Returns the most-likely sequence
    """
    len_ = len(obs)
    # stores p(best_seqence)
    prob = [[[0] for i in range(states)] for j in range(len_)]
    # stores most-likely sequence
    seq = [[[''] for i in range(states)] for i in range(len_)]
    # initializes uniform state distribution
    prob[0] = [staterow[obs[0]] / len(A) for staterow in O]
    # initialize best sequence of length 1
    seq[0] = [str(i) for i in range(states)]
    # We iterate through all indices in the data
    for length in range(1, len_):   # length + 1 to avoid initial condition
        for state in range(states):
            max_state = 0
            best_prob = 0
            # We iterate through all possible transitions from previous state
            for prev in range(states):
                # cur_prob is the probability of transitioning to 'state'
                # from 'prev' state and observing the correct state.
                cur_prob = prob[length - 1][prev] * A[prev][state] *\
                    O[state][obs[length]]
                if cur_prob > best_prob:
                    max_state, best_prob = prev, cur_prob
                # update best probability
                prob[length][state] = best_prob
                # update sequence
                seq[length][state] = seq[length - 1][max_state] + str(state)
        prob[length] = prob[length][:]   # copies by value
        seq[length] = seq[length][:]
    max_ind = 0
    for i in range(states):  # find most-likely index of entire sequence
        if prob[len_ - 1][i] > prob[len_ - 1][max_ind]:
            max_ind = i
    # returns most-likely sequence
    return seq[len_ - 1][max_ind]


def forward(obs, A, O, pi):
    """Computes the probability a given HMM emits a given observation using the
        forward algorithm. This uses a dynamic programming approach, and uses
        the 'prob' matrix to store the probability of the sequence at each length.
        Arguments: num_states the number of states
                   obs        an array of observations
                   A          the transition matrix
                   O          the observation matrix
    """
    len_ = len(obs)                   # number of observations
    num_states = pi.shape[0]
    alpha = np.zeros((len_, num_states))
    # Base case
    alpha[0, :] = pi * O[:,obs[0]]
    probability = 1
    C_normalize = sum(alpha[0, :])
    if C_normalize != 0:
        alpha[0, :] = alpha[0, :] / C_normalize
        probability *= C_normalize

    # We iterate through all indices in the data and use dynamic programming to update
    for length in range(1, len_):   # length + 1 to avoid initial condition
        #for state in range(num_states):
            #for prev_state in range(num_states):
                #alpha[length, state] += alpha[length-1, prev_state] * A[prev_state, state] * O[state, obs[length]]
        alpha[length, :] = np.dot(alpha[length-1, :], A) * O[:,obs[length]]

        # Normalize to prevent underflow 
        C_normalize = sum(alpha[length, :])
        if C_normalize != 0:
            alpha[length, :] = alpha[length, :] / C_normalize
        probability *= C_normalize

    # return total probability
    return (alpha,probability) 
    # Unnormalized case
    #return (alpha,sum(alpha[len_-1,:])) 

def backward(obs, A, O, pi):
    """ Computes the probability a given HMM emits a given oservation using the
        backward algorithm. This uses a dynamic programming approach
    """
    len_ = len(obs)
    num_states = pi.shape[0]
    
    beta = np.zeros((len_, num_states))
    # Base case - last beta is 1
    beta[len_-1, :] = 1.0
    C_normalize = sum(beta[len_-1, :])
    if C_normalize != 0:
        beta[len_-1, :] = beta[len_-1, :] / C_normalize
        probability = C_normalize
    
    # Calculate rest of beta
    for i in range(len_-2, -1, -1):
        #for state in range(num_states):
            #for next_state in range(num_states):
                #beta[i, state] += beta[i+1,next_state] * A[state, next_state] * O[next_state, obs[i+1]]
        beta[i, :] = np.dot(A, (O[:,obs[i+1]]*beta[i+1,:]))

        # Normalize to prevent underflow 
        C_normalize = sum(beta[i, :])
        if C_normalize != 0:
            beta[i, :] = beta[i, :] / C_normalize
            probability *= C_normalize
    
    # Return total probability
    return (beta, probability*np.sum(pi * O[:, obs[0]] * beta[0,:]))
    # Unnormalized case
    #return (beta, np.sum(pi * O[:, obs[0]] * beta[0,:]))


def baum_welch(training, A, O, pi, iterations):
    A, O, pi = np.copy(A), np.copy(O), np.copy(pi)
    num_states = pi.shape[0]
    num_words = O.shape[1]
    print O.shape
    step = 0
    norm_diff = 1

    while norm_diff > 1e-8*num_words*num_states and step < iterations:
        print step
        step += 1
        A1 = np.zeros_like(A)
        O1 = np.zeros_like(O)
        pi1 = np.zeros_like(pi)

        for obs in training:
            # E-step - Compute forward-backward
            alpha, za = forward(obs, A, O, pi)
            beta, zb = backward(obs, A, O, pi)
            #print za
            #assert abs(za - zb) <1e-6, "marginals not equal"

            # M-step - maximum likelihood estimate
            pi1 += alpha[0,:] * beta[0,:]
            for i in range(0, len(obs)):
                O1[:, obs[i]] += alpha[i,:] * beta[i,:]
            """for i in range(1, len(obs)):
                for s1 in range(num_states):
                    for s2 in range(num_states):
                        A1[s1,s2] += alpha[i-1,s1]*A[s1,s2]*O[s2,obs[i]]*beta[i,s2]"""
            xi = np.zeros((num_states,num_states,len(obs)-1));
            for t in range(len(obs)-1):
                denom = np.dot(np.dot(alpha[t, :], A) * O[:,obs[t+1]].T,beta[t+1,:].T)
                for i in range(num_states):
                    numer = alpha[t,i] * A[i,:] * O[:,obs[t+1]].T * beta[t+1,:]
                    xi[i,:,t] = numer / denom
  
            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.squeeze(np.sum(xi,axis=1))
            # Need final gamma element for new B
            prod =  (alpha[len(obs)-1,:] * beta[len(obs)-1,:]).reshape((-1,1))
            gamma = np.hstack((gamma,  prod / np.sum(prod))) #append one more to gamma!!!

            #newpi += gamma[:,0]
            A1 += np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
            """newO = np.copy(O)
            numLevels = O.shape[1]
            sumgamma = np.sum(gamma,axis=1)
            gamma = np.array(gamma)
            for lev in range(numLevels):
                mask = obs == lev
                try:
                    newO[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma
                except ValueError:
                    pass"""


        # Normalize
        pi = pi1 / np.sum(pi1)
        for s in range(num_states):
            A1[s, :] = A1[s,:] / np.sum(A1[s,:])
            O1[s, :] = O1[s, :] / np.sum(O1[s, :])
        # Take advantage of spareness by rounding to zero
        if step % 25 == 0:
            A1 = A1 * (A1 > 1e-10)
            pi = pi * (pi > 1e-10)
        #print A1
        #print newA
        norm_diff = LA.norm(A1-A) + LA.norm(O1-O)
        print norm_diff
        A[:] = A1
        O[:] = O1
    return pi, A, O

# Generating two dictionaries (we probably only need one but oh well).
def generateMaps(sonnets):
    wordMap = {}
    intMap = {}
    wordCount = {}
    counter = 0
    setOfWords = []

    for sonnet in sonnets:
        for word in sonnet:
            word = word.strip()
            if word not in setOfWords:
                wordMap[word] = counter
                intMap[counter] = word
                wordCount[word] = 1
                counter += 1
                setOfWords.append(word)
            else:
                wordCount[word] += 1

    return (wordMap, intMap, wordCount)

# Mapping the string words into integers. This is how we will tokenize things.
def mapWordToInt(sonnets, wordMap):
    newDataSet = []

    for sonnet in sonnets:
        intSonnet = []
        for word in sonnet:
            intRepresentation = wordMap[word]
            intSonnet.append(intRepresentation)
        newDataSet.append(intSonnet)
    return newDataSet

# Mapping integers to word. This is how we find out our poem.
def mapIntToWord(line, intMap):
    lineTransp = line.T.tolist()
    print lineTransp
    oneLine = []
    for word in lineTransp[0]:
        wordRepresentation = intMap[word]
        oneLine.append(wordRepresentation)
    return oneLine

# Extracting data here.
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

# This will assume uniform probability initial state.
def generateStartProb(numStates):
    initProb = 1 / float(numStates)
    pi = np.array([initProb for i in range(numStates)])
    return pi

def analyzeHiddenStates(O, wordMap, intMap, wordCount):
    """ This function finds the top ten words
    in the hidden states """
    for i in intMap:
        O[:, i] = O[:,i]/sum(O[:,i])
    Ot = O.transpose()
    best = []
    cur_best = []
    for i in O:
        cur_best_pos = []
        cur_best_probs = heapq.nlargest(20, i)
        for j in cur_best_probs:
            for row, prob in enumerate(i):
                if prob == j:
                    cur_best_pos.append(row)
        cur_best = [intMap[j] for j in cur_best_pos]
        best.append(cur_best)
    return best


def simulate(nSteps, A, O, pi):
    # For testing
    def drawFrom(probs):
        return np.where(np.random.multinomial(1,probs) == 1)[0][0]
    observations = np.zeros(nSteps)
    states = np.zeros(nSteps)
    states[0] = drawFrom(pi)
    observations[0] = int(drawFrom(O[states[0],:]))
    for t in range(1, nSteps):
        states[t] = drawFrom(A[states[t-1],:])
        observations[t] = int(drawFrom(O[states[t],:]))
    return observations, states


if __name__ == '__main__':
    #test()
    #test2()
    #test_file()
    np.random.seed(13)
    random.seed(13)
    main()
    generate()

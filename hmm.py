# Based on solution to HW4 provided by Fabian Boemer and Kevin Tang
import sys
import numpy as np
import random
# for frobenius norm
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from itertools import chain
import heapq

def main():
    #trainingWords = getData("complete_shakespeare_words.txt")
    trainingWords = getData("shakespeareWords.txt")
    wordMap, intMap, wordCount = generateMaps(trainingWords)

    trainingSequence = mapWordToInt(trainingWords, wordMap)
    numObs = len(wordMap)

    # A, O are randomly initialized based on the number of states
    # and observations.

    H_STATES = 8
    A, O = randomlyInitialize(H_STATES, numObs)
    pi = generateStartProb(H_STATES)

    # Now, going to try to run baum_welch

    trainedPi, trainedA, trainedO = baum_welch(trainingSequence, A, O, pi, 500)

    # Save matrices to file
    writeHMM('test{}.txt'.format(H_STATES), trainedA, trainedO, trainedPi)

    good_words = analyzeHiddenStates(O, wordMap, intMap, wordCount)
    print good_words


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
    pi = np.array([0.5, 0.5])
    A = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    O = np.array([[0.3, 0.3, 0.4],
                  [0.2, 0.5, 0.3]])
    print baum_welch([observ], A, O, pi, 50)

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
        for state in range(num_states):
            for prev_state in range(num_states):
                alpha[length, state] += alpha[length-1, prev_state] * A[prev_state, state] * O[state, obs[length]]
    

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
        for state in range(num_states):
            for next_state in range(num_states):
                beta[i, state] += beta[i+1,next_state] * A[state, next_state] * O[next_state, obs[i+1]]
        
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
    step = 0
    norm_diff = 1

    while norm_diff > 1e-4 and step < iterations:
        print step
        step += 1
        A1 = np.zeros_like(A)
        O1 = np.zeros_like(O)
        pi1 = np.zeros_like(pi)

        for obs in training:
            # E-step - Compute forward-backward
            alpha, za = forward(obs, A, O, pi)
            beta, zb = backward(obs, A, O, pi)
            assert abs(za - zb) <1e-6, "marginals not equal"

            # M-step - maximum likelihood estimate
            pi1 += alpha[0,:] * beta[0,:]
            for i in range(0, len(obs)):
                O1[:, obs[i]] += alpha[i,:] * beta[i,:]
            for i in range(1, len(obs)):
                for s1 in range(num_states):
                    for s2 in range(num_states):
                        A1[s1,s2] += alpha[i-1,s1]*A[s1,s2]*O[s2,obs[i]]*beta[i,s2]

        # Normalize
        pi = pi1 / np.sum(pi1)
        for s in range(num_states):
            A1[s, :] = A1[s,:] / np.sum(A1[s,:])
            O1[s, :] = O1[s, :] / np.sum(O1[s, :])
        norm_diff = LA.norm(A1-A) + LA.norm(O1-O)
        print norm_diff
        A = A1
        O = O1
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
        O[:, i] = O[:,i] / wordCount[intMap[i]]
    Ot = O.transpose()
    best = []
    cur_best = []
    for i in O:
        cur_best_pos = []
        cur_best_probs = heapq.nlargest(10, i)
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
        states[0] = drawFrom(A[states[0],:])
        observations[t] = int(drawFrom(O[states[t],:]))
    return observations, states


if __name__ == '__main__':
    np.random.seed(13)
    test2()
    #test_file()
    #main()

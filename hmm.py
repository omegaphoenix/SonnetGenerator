# Based on solution to HW4 provided by Fabian Boemer and Kevin Tang
import numpy as np

def main():

def loadHMM(filename):
    """ Loads a HMM file. Returns in format A, O, sequences
    """
    A = []  # transition matrix
    O = []  # observation matrix
    sequences = []

    with open(filename, 'r') as f:
        num_states, num_obs = [int(x)
                               for x in f.readline().strip().split('\t')]
        for i in range(num_states):
            A.append([float(x) for x in f.readline().strip().split('\t')])
        for i in range(num_states):
            O.append([float(x) for x in f.readline().strip().split('\t')])
        for i in range(5):
            sequences.append([int(x) for x in list(f.readline().strip())])
    return (A, O, sequences)

def randomlyInitialize():
    pass


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


def forward(num_states, obs, A, O, pi):
    """Computes the probability a given HMM emits a given observation using the
        forward algorithm. This uses a dynamic programming approach, and uses
        the 'prob' matrix to store the probability of the sequence at each length.
        Arguments: num_states the number of states
                   obs        an array of observations
                   A          the transition matrix
                   O          the observation matrix
    """
    len_ = len(obs)                   # number of observations
    # stores p(seqence)
    #prob = [[[0.] for i in range(num_states)] for i in range(len_)]
    alpha = np.zeros((len_, num_states))

    # initializes uniform state distribution, factored by the
    # probability of observing the sequence from the state (given by the
    # observation matrix)
    #prob[0] = [(1. / num_states) * O[j][obs[0]] for j in range(num_states)]
    alpha[0, :] = pi * O[:,observations[0]]

    # We iterate through all indices in the data
    for length in range(1, len_):   # length + 1 to avoid initial condition
        for state in range(num_states):
            # stores the probability of transitioning to 'state'
            #p_trans = 0

            # probabilty of observing data in our given 'state'
            #p_obs = O[state][obs[length]]

            # We iterate through all possible previous states, and update
            # p_trans accordingly.
            for prev_state in range(num_states):
                #p_trans += prob[length - 1][prev_state] * A[prev_state][state]
                alpha[length, state] += alpha[length-1, prev_state] * a[prev_state, state] * O[state, observations[i]]

            #prob[length][state] = p_trans * p_obs  # update probability

        #prob[length] = prob[length][:]  # copies by value

    # return total probability
    #return sum(prob[len_ - 1])
    return (alpha, np.sum(alpha[N-1,:]))

def backward(num_states, obs, A, O, pi):
    """ Computes the probability a given HMM emits a given oservation using the
        backward algorithm. This uses a dynamic programming approach
    """
    len_ = len(obs)
    beta = np.zeros((len_, num_states))
    # Base case - last beta is 1
    beta[len_-1, :] = 1
    # Calculate rest of beta
    for i in range(len_-2, -1, -1):
        for s1 in range(num_states):
            for s2 in range(num_states):
                beta[i, s1] += beta[i+1,s2] * A[s1, s2] * O[s2, observations[i+1]]

    return (beta, np.sum(pi * O[:, obs[0]]*beta[0,:]))


def baum_welch(training, A, O, pi, iterations):
    A, O, pi = np.copy(A), np.copy(O), np.copy(pi)
    num_states = pi.shape[0]

    for step in range(iterations):
        Al = np.zeros_like(A)
        O1 = np.zeros_like(O)
        pi1 = np.zeros_like(pi)

        for obs in training:
            # Compute forward-backward
            alpha, za = forward(num_states, obs, A, O, pi)
            beta, zb = backward(num_states, obs, A, O, pi)
            assert abs(za - zb) <1e-6, "marginals don't agree"

            # M-step
            pi1 += alpha[0,:] * beta[0,:] / za
            for i in range(0, len(obs)):
                O1[:, observations[i]] += alpha[i,:] * beta[i,:] / za
            for i in range(1, len(obs)):
                for s1 in range(S):
                    for s2 in range(S):
                        A1[s1,s2] += alpha[i-1,s1]*A[s1,s2]*O[s2,obs[i]]*beta[i,s2]/za
        # normalize
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            A[s, :] = Al[s,:] / np.sum(Al[s,:])
            O[s, :] = O1[s, :] / np.sum(O1[s, :])
    return pi, A, O

if __name__ == '__main__':
    main()

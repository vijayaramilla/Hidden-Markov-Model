import numpy as np


def forward(x, pi, A, B):
    """ Run the forward algorithm for a single example.

    Args:
        x: A 1-D int NumPy array with shape [T], where each element
            is either 0, 1, 2, ..., or N_x - 1. T is the length of
            the observation sequence and N_x is the number of possible
            values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        alpha, a 2-D float NumPy array with shape [T, N_z].
    """


    alpha = np.zeros(shape=(x.shape[0],A.shape[0]))


    alpha[0][:]=pi*B[:,x[0]]
    for i in range(1,len(x)):
        alpha[i][:] = B[:,x[i]] * np.dot(A.transpose(), alpha[i-1][:])

    return alpha


def backward(x, pi, A, B):
    """ Run the backward algorithm for a single example.

    Args:
        x: A 1-D int NumPy array with shape [T], where each element
            is either 0, 1, 2, ..., or N_x - 1. T is the length of
            the observation sequence and N_x is the number of possible
            values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        beta, a 2-D float NumPy array with shape [T, N_z].
    """
    # TODO: Write this function.

    beta = np.zeros(shape=(x.shape[0],A.shape[0]))
    beta[len(x)-1][:]=np.ones(B.shape[0])
    # print(beta)
    last = len(x)-1
    # print(last)
    for i in range(last,0,-1):
        beta[i-1]=  np.dot(A, B[:,x[i]] * beta[i])
    return beta

def individually_most_likely_states(X, pi, A, B):
    """ Computes individually most-likely states.

    By "individually most-likely states," we mean that the *marginal*
    distributions are maximized. In other words, for any particular
    time step of any particular sequence, each returned state i is
    chosen to maximize P(z_t = i | x).

    All sequences in X are assumed to have the same length, T.

    Args:
        X: A 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., or N_x - 1. N is the number of observation
            sequences, T is the length of every sequence, and N_x is the number
            of possible values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        Z, a 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., N_z - 1.
    """


    Z = np.zeros(shape=(X.shape),dtype=int)

    for i in range(X.shape[0]):

        alpha = forward(X[:][i], pi, A, B)
        beta = backward(X[:][i], pi, A, B)
        p_x = alpha[-1].sum()
        marginal = (alpha * beta)/p_x


        margin=np.argmax(marginal,1)
        Z[i][:]=margin
    return Z




def take_EM_step(X, pi, A, B):
    """ Take a single expectation-maximization step.

    Args:
        X: A 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., or N_x - 1. N is the number of observation
            sequences, T is the length of every sequence, and N_x is the number
            of possible values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        A tuple containing
        pi_prime: pi after the EM update.
        A_prime: A after the EM update.
        B_prime: B after the EM update.
    """
    # TODO: Write this function.

    pi_dash = np.zeros(shape=pi.shape)
    A_dash = np.zeros(shape=A.shape)
    B_dash = np.zeros(shape=B.shape)

    for i in range(X.shape[0]):

        alpha = forward(X[:][i], pi, A, B)
        beta = backward(X[:][i], pi, A, B)
        x=X[:][i]

        p_x = alpha[-1].sum()
        pi_dash+=(alpha[0]*beta[0])/p_x

        b_empty=np.zeros(shape=B.shape)

        for t in range(len(alpha)):
            curr_alpha = alpha[t]
            curr_beta =beta[t]
            curr_alpha_beta = curr_alpha*curr_beta

            b_empty[:,x[t]]+=curr_alpha_beta

        curr_A = np.zeros((A.shape))
        for t in range(len(alpha) - 1):
            alpha__ = alpha[t]
            b = B[:, x[t + 1]]
            curr_A += (A * b * beta[t + 1]) * alpha__[:, np.newaxis]

        A_dash += (curr_A /p_x)
        B_dash+=b_empty/p_x


    pi_dash=pi_dash/np.sum(pi_dash)



    row_sums = B_dash.sum(axis=1)
    B_dash = B_dash / row_sums[:, np.newaxis]

    row_sums = A_dash.sum(axis=1)
    A_dash = A_dash / row_sums[:, np.newaxis]


    return pi_dash,A_dash,B_dash
    # raise NotImplementedError('Not yet implemented.')

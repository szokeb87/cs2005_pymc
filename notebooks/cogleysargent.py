import numpy as np
import scipy as sp
from scipy.linalg import inv

import pymc as pm
from numba import jit
from math import ceil

def B_tril(b):
    """_
    This function builds a lower triangular matrix from the elements of b (array), so
    that the diagonal elements are all 1s (corresponds to the B matrix in the paper).
    """
    bb, place = b.tolist(), 0
    K = int(ceil(np.sqrt(len(bb) * 2)))
    if K > 0:                 # if b contains any element
        bb.insert(place, 1.)
        for ii in range(1, K):
            place += ii + 1
            bb.insert(place, 1.)
        B, indices = np.eye(K), np.tril_indices(K)
        B[indices] = bb
        return B
    else:                   # if b is empty it gives back B=1
        return np.asarray([[1]])


def order_list(self, stochastic):
    """
    Because StepMethod stores collections of stochastic variables (e.g. Theta, LH, etc) as
    an unordered set in self.stochastics, we need to use a trick to save the ordering
    in the list: self.my_list
    """
    x_dict = {x:y for y, x in enumerate(stochastic)}

    # Since we subclass StepMethod directly, the result of the line below is to create
    # self.stochastics (notice the plural form!) which is a set of stochastic variables
    pm.StepMethod.__init__(self, stochastic)

    my_list = [None]*len(self.stochastics)
    for element in self.stochastics:
        my_list[x_dict[element]] = element

    return my_list


class ForwardBackward(pm.StepMethod):

    """
    This Step Method implements the forward backward recursion by Carter and Kohn (1994).

    Updates:

        Theta: pymc.Container
               constructed from a list of (T+1) ordered stochastic (KM-)arrays. This is theta^T
               and it is updated jointly.

    Other variables:

        Y:     pymc.Container
               constructed from an ordered list of T stochastic (K-)arrays with fixed values
               (observed stochastic variables). This is y^T.

        R:     pymc.Container
               constructed from an ordered list of T deterministic (KxK-)arrays. It contains all
               R_t whose parents are Binv (another determinisitc variable, children of Betas) and
               LH, i.e. it summarizes the current values of Betas and H^T.

        Q_inv: pymc.stochastic variable
               Inverse of the covariance of theta_{t+1} conditional on theta_t. Since it is the
               inverse of Q, we model it as a Wishart stochastic variable (it allows us to use
               less matrix inversion, thus making the code faster)

        X_obs: numpy.array
               of the RHS variables from the measurement equation. Its shape must be T x M and
               its first column must be all 1s.
    """

    def __init__(self, stochastic, Y, R, Q_inv, X_obs):

        # Create an ORDERED list from stochastics (= set)
        self.my_list = order_list(self, stochastic)

        # Data (observed stochastic variables)
        self.Y, self.X = Y, X_obs

        # Size of numpy arrays
        self.T = np.asarray(self.Y.value).shape[0]
        self.K = np.asarray(self.Y.value).shape[1]                  # number of equations in the VAR
        self.KM = X_obs.shape[0]
        self.M = self.KM / self.K                                   # M = (1 + K*L)

        # Other non-observed stochastic variables
        self.Qinv = Q_inv
        self.R = R

        # Initial moments for the prior
        self.theta0 = self.my_list[0].parents['mu']
        self.P0 = self.my_list[0].parents['C']

        # Auxiliary matrix for the rejection sampling (to construct a test matrix A)
        self.L = (self.M - 1) / self.K
        self.S = np.hstack([np.eye(self.K * (self.L - 1)),
                            np.zeros((self.K * (self.L - 1), self.K))])


    def propose(self):

        Q = np.linalg.inv(self.Qinv.value)
        R = np.asarray(self.R.value)                                # 3-dimensional: T x K x K
        y = np.asarray(self.Y.value)
        theta = forwardbackward_jit(y, self.X, Q, R, self.theta0, self.P0)

        # Assigning new values
        # remember: my_list is a sorted list starting with Theta_0, Theta_1, etc..
        for ind, stoch in enumerate(self.my_list):
            stoch.value = theta[:, ind]

    #-----------------------------------------------------
    # REJECTION SAMPLING:
    #-----------------------------------------------------
    def step(self):

        # Draw a proposal
        self.propose()

        # Reject the proposal if any element of Theta violates the stability condition
        for ss in self.stochastics:
            theta_test = ss.value
            A = [theta_test[1:self.M].tolist()]
            for i in range(1, self.K):
                A.append(theta_test[1 + i * self.M : (i+1) * self.M].tolist())
            A = np.vstack((np.asarray(A), self.S))
            if abs(sp.linalg.eig(A)[0]).max() > 1.:
                self.reject()

    def reject(self):
        # Sets current s value to the last accepted value (stochastic.value = stochastic.last_value)
        for stoch in self.stochastics:
            stoch.revert()



class W_Q(pm.Gibbs):

    """
    This step method (subclass of the Gibbs StepMethod) updates Qinv (Wishart + Normal -> Wishart)

    Updates:

        Q_inv: pymc.stochastic variable
               Inverse of the covariance of theta_{t+1} conditional on theta_t. Since it is the
               inverse of Q, we model it as a Wishart stochastic variable (it allows us to use
               less matrix inversion, thus making the code faster)

    Other variables:

        Theta: pymc.Container
               constructed from a list of (T+1) ordered stochastic (KM-)arrays. This is theta^T
               and it is updated jointly.
    """

    def __init__(self, stochastic, Theta):
        pm.Gibbs.__init__(self, stochastic)
        self.conjugate = True

        # Other non-observed stochastic variables
        self.Theta = Theta

        # Initial moments for the prior
        self.df = stochastic.parents["n"]
        self.Q_bar = stochastic.parents["Tau"]


    def propose(self):
        theta = np.asarray(self.Theta.value)
        TT = theta.shape[0]
        v = np.diff(theta, axis = 0)

        self.stochastic.value = pm.rwishart(self.df + TT, self.Q_bar + v.T @ v)


class IG_Sigma(pm.StepMethod):

    """
    This Step Method updates all sigma_i^2 for i=1,...,K

    Updates:

        Sigma2: pymc.Container
                constructed from a K-list of stochastic variables, i.e. [sigma_1^2,...,sigma^2_K].
                Notice that this is the variance (not the standard deviation!) of log(h_{t+1})
                conditional on log(h_t). They are updated jointly, but their independence is taken
                into account -> covariance matrix of log(h_{t+1}) is np.diag( Sigma2 )

    Other variables:

        LH:     pymc.Container
                constructed from an ordered list of T+1 stochastic (K-)arrays. These are log(h_t),
                where h_t = [h_{1,t},...,h_{K,t}]. Notice that it contains the LOG of h_t, i.e. log( H^T ).
    """


    def __init__(self, stochastic, LH):

        self.my_list = order_list(self, stochastic)

        # Other non-observed stochastic variables
        self.LH = LH

        # Initial moments for the prior
        a0 = np.empty(len(self.my_list))
        b0 = np.empty(len(self.my_list))

        for ind, stoch in enumerate(self.my_list):
            a0[ind] = stoch.parents["alpha"]
            b0[ind] = stoch.parents["beta"]
        self.a0 = a0
        self.b0 = b0

    def step(self):
        dlnh = np.diff(np.asarray(self.LH.value), axis = 0)
        TT = dlnh.shape[0]
        a1 = self.a0 + TT
        b1 = self.b0 + np.sum(dlnh**2, 0)

        for ind, stoch in enumerate(self.my_list):
            stoch.value = pm.rinverse_gamma(a1[ind]/2, b1[ind]/2)




class N_Beta(pm.Gibbs):

    """
    This step method (subclass of the Gibbs StepMethod) updates Betas, a vector of parameters in B.

    Updates:

        Betas: pymc.stochastic variable
               Stochastic (J-)array containing beta_{2,1},...,beta_{K,K-1}. The function B_tril
               turns this array into a lower triangular matrix B (see the paper).

    Other variables:

        Y:     pymc.Container
               constructed from an ordered list of T stochastic (K-)arrays with fixed values
               (observed stochastic variables). This is y^T.

        mu_Y:  pymc.Container
               constructed from an ordered list of T deterministic (K-)arrays. It contains
               E_t[y_{t+1}] = X_t * theta_t for all t. Their parents are X_obs (data) and
               Theta, i.e. it concisely embodies the data and the current values of theta^T.

        LH:    pymc.Container
               constructed from an ordered list of T+1 stochastic (K-)arrays. These are log(h_t),
               where h_t = [h_{1,t},...,h_{K,t}]. Notice that it contains the LOG of h_t, i.e. LH is
               log( H^T ) and not H^T.
    """

    def __init__(self, stochastic, Y, mu_Y, LH):
        pm.Gibbs.__init__(self, stochastic)
        self.conjugate = True   # pymc will include a Metropolis rejection step if this is false

        # Data (observed stochastic variables)
        self.Y = Y

        # Other non-observed determinisitc and stochastic variables
        self.muY = mu_Y
        self.LH = LH

        # Initial moments for the prior
        self.b_bar = self.stochastic.parents["mu"]
        self.Pb_bar = self.stochastic.parents["C"]

    def propose(self):
        ee = np.asarray(self.Y.value) - np.asarray(self.muY.value)
        H = (np.exp(self.LH.value)**(-0.5))[1:]
        K = np.asarray(self.Y.value).shape[1]
        b_new = np.empty_like(self.stochastic.value)

        # auxiliary variables to pick the right subvector/submatrix for the equations
        lb = 0
        ub = 1

        for j in range(1, K):
            z = np.expand_dims(H[:, j], 1)*np.expand_dims(ee[:, j], 1)     # LHS variable in the regression
            Z = np.expand_dims(-H[:, j], 1)*ee[:, :j]                      # RHS variables in the regression

            b_prior = np.asarray([self.b_bar[lb:ub]])
            Vinv_prior = inv(self.Pb_bar[lb:ub, lb:ub])

            V_post = inv(Vinv_prior + Z.T @ Z)
            b_post = V_post @ (Vinv_prior @ b_prior.T + Z.T @ z)

            b_new[lb:ub] = pm.rmv_normal_cov(b_post.ravel(), V_post)
            lb = ub
            ub += j+1

        self.stochastic.value = b_new




class Metropolis_LH(pm.StepMethod):

    """
    This Step Method updates the matrix log( H^T ) of stochastic volatilities following the univariate
    algorithm of Jacquier et al. (1994).

    Updates:

        LH:     pymc.Container
                constructed from an ordered list of T+1 stochastic (K-)arrays. These are log(h_t),
                where h_t = [h_{1,t},...,h_{K,t}]. Notice that it contains the LOG of h_t, i.e. log( H^T ).

    Other variables:

        Y:     pymc.Container
               constructed from an ordered list of T stochastic (K-)arrays with fixed values
               (observed stochastic variables). This is y^T.

        mu_Y:  pymc.Container
               constructed from an ordered list of T deterministic (K-)arrays. It contains
               E_t[y_{t+1}] = X_t * theta_t for all t. Their parents are X_obs (data) and
               Theta, i.e. it concisely embodies the data and the current values of theta^T.

        Betas: pymc.stochastic variable
               Stochastic (J-)array containing beta_{2,1},...,beta_{K,K-1}. The function B_tril
               turns this array into a lower triangular matrix B (see the paper).

        Sigma2: pymc.Container
                constructed from a K-list of stochastic variables, i.e. [sigma_1^2,...,sigma^2_K].
                Notice that this is the variance (not the standard deviation!) of log(h_{t+1})
                conditional on log(h_t). They are updated jointly, but their independence is taken
                into account -> covariance matrix of log(h_{t+1}) is np.diag( Sigma2 )

    """

    def __init__(self, stochastic, Y, mu_Y, Betas, Sigma2):

        self.my_list = pm.Container(order_list(self, stochastic))

        # Data (observed stochastic variables)
        self.Y = Y
        self.T = np.asarray(self.Y.value).shape[0]        # number of observations
        self.K = np.asarray(self.Y.value).shape[1]        # number of equations in the VAR

        # Other non-observed determinisitc and stochastic variables
        self.muY = mu_Y
        self.Betas = Betas
        self.Sigma2 = Sigma2

        # Initial moments for the prior
        self.lnh_bar = self.my_list[0].parents["mu"]
        self.Ph_bar = self.my_list[0].parents["C"]

    def propose(self):

        sig2  = np.asarray(self.Sigma2.value)
        y = np.asarray(self.Y.value)
        ee = np.asarray(self.Y.value) - np.asarray(self.muY.value)
        BB = B_tril(self.Betas.value)
        ut = (BB @ ee.T).T
        lnH_old = np.asarray(self.my_list.value)

        lnH_new = metropolis_lh_jit(lnH_old, y, ut, sig2, self.lnh_bar, self.Ph_bar)

        #--------------------------------------
        # Assigning new values
        #--------------------------------------
        for ind, stoch in enumerate(self.my_list):
            stoch.value = lnH_new[ind, :]

    def step(self):
        self.propose()

####
#
####

@jit(nopython = True)
def forwardbackward_jit(y, X, Q, R, theta0, P0):

    # Size of numpy arrays
    T = y.shape[0]
    N = y.shape[1]                               # number of equations in the VAR
    M = int(X.shape[0]/N)                        # M = (1 + K*L)
    KM = N * M

    P = np.empty((T + 1, KM, KM))                # P_{t-1|t-1}
    P_ = np.empty((T, KM, KM))                   # P_{t|t-1}
    Th = np.empty((KM, T + 1))                   # Theta_{t-1|t-1}
    theta = np.empty((KM, T + 1))                # container for new draws

    K = np.empty((KM, N))

    P[0, :, :] = P0
    Th[:, 0] = theta0

    #-----------------------------------------------------
    # FORWARD STEP (Kalman filter):
    #-----------------------------------------------------
    for tt in range(T):
        XX = X[:, :, tt].T
        P_[tt, :, :] = P[tt, :, :] + Q
        K[:, :] = (P_[tt, :, :] @ XX.T) @ np.linalg.inv(XX @ P_[tt, :, :] @ XX.T + R[tt])

        Th[:, tt + 1] = Th[:, tt] + K @ (y[tt, :].T - XX @ Th[:, tt])
        P[tt + 1, :, :] = P_[tt, :, :] - K @ XX @ P_[tt, :, :]

    #-----------------------------------------------------
    # BACKWARD STEP:
    #-----------------------------------------------------
    theta[:, -1] = Th[:, -1] + np.linalg.cholesky(P[-1, :, :]) @ np.random.randn(KM)

    for tt in range(2, T + 2):
        back_mean = Th[:,-tt] + P[-tt, :, :] @ np.linalg.inv(P_[-(tt-1), :, :]) @ (theta[:, -(tt-1)] - Th[:, -tt])
        back_cov = P[-tt, :, :] - P[-tt, :, :] @ np.linalg.inv(P_[-(tt-1), :, :]) @ P[-tt, :, :]
        theta[:, -tt] = back_mean + np.linalg.cholesky(back_cov) @ np.random.randn(KM)

    return theta



@jit(nopython = True)
def metropolis_lh_jit(lnH_old, Y, ut, sig2, lnh_bar, Ph_bar):

    T, K = Y.shape        # numb of observations, numb of equations in the VAR
    lnH_new = np.empty_like(lnH_old)


    for i in range(K):
        #--------------------------------------
        # update the initial log(h_0)
        #--------------------------------------
        ss0 = Ph_bar[i, i]
        ss = ss0 * sig2[i] /(sig2[i] + ss0)
        lnh0_mu = ss*(lnh_bar[i]/ss0 + lnH_old[1, i]/sig2[i] )

        lnH_new[0, i] = lnh0_mu + np.sqrt(ss)*np.random.randn()

        #--------------------------------------
        # update log(h_t) step-by-step for 0<t<T
        #--------------------------------------
        for tt in range(1, T):
            # mean and variance for log(h) (proposal density)
            lnh_mu = (.5) * (lnH_new[tt - 1, i] + lnH_old[tt + 1, i])
            # candidate draw from normal
            lnh_prop = lnh_mu + np.sqrt((.5)*sig2[i])*np.random.randn()

            # acceptance probability
            lp1 = -0.5 * lnh_prop - ut[tt - 1, i]**2/(2 * np.exp(lnh_prop))
            lp0 = -0.5 * lnH_old[tt, i] - ut[tt - 1, i]**2/(2 * np.exp(lnH_old[tt, i]))
            accept = np.minimum(1., np.exp(lp1 - lp0))

            u = np.random.rand()
            if u <= accept:
                lnH_new[tt, i] = lnh_prop
            else:
                lnH_new[tt, i] = lnH_old[tt, i]

        #--------------------------------------
        # update the last h_T
        #--------------------------------------
        # mean and variance for log(h) (proposal density)
        lnh_mu = lnH_new[T-1,i]
        # candidate draw from normal
        lnh_prop = lnh_mu + np.sqrt(sig2[i])*np.random.randn()

        # acceptance probability
        lp1 = -0.5*lnh_prop - ut[-1, i]**2/(2*np.exp(lnh_prop))
        lp0 = -0.5*lnH_old[T, i] - ut[-1, i]**2/(2*np.exp(lnH_old[T, i]))
        accept = np.minimum(1., np.exp(lp1 - lp0))

        u = np.random.rand()
        if u <= accept:
            lnH_new[T, i] = lnh_prop
        else:
            lnH_new[T, i] = lnH_old[T, i]

    return lnH_new

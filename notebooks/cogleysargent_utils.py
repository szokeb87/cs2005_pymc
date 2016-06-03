import numpy as np
import scipy as sp
from scipy.linalg import inv

import pymc as pm
from math import ceil
from cogleysargent import *


class Generate_prior(object):
    """
    Following the approach of the paper, this class determines parameter values for the
    pre-specified prior distributions by estimating a SUR model on the training sample
    (which consists of the first training_years years of the original sample)

    """

    def __init__(self, data, lags = 2, training_years = 11):
        self.data = np.atleast_2d(data)
        self.T, self.N = self.data.shape      # T = number of periods, N = number of equations
        self.L = lags                         # VAR lag order, in the paper: VAR(2)
        self.numb_obs = self.T - self.L

        self.T0 = 4 * training_years - self.L + 1
        if self.T0 > self.L:
            self.T_training = self.T0 - self.L
        else:
            self.T_training = 0

    def create_YX(self):
        """
        This function creates the data matrices for the SUR
        """

        # Pull out necessary parameters
        y = self.data
        N, L = self.N, self.L
        numb_obs = self.numb_obs
        T_training = self.T_training

        Y = y[L:, :].T
        X = np.empty((N*(1 + N * L), N, numb_obs))

        # lagged dependent variables: X1 = [y(t-1) y(t-1) ... y(t-L)]
        X1 = np.ones((numb_obs, 1 + (N * L)))
        for i in range(1, L + 1):
            cols = np.arange(1 + N * (i - 1), 1 + (i * N))
            X1[:, cols] = y[L - i:-i, :]

        for tt in range(numb_obs):
            X[:, :, tt] = np.kron(np.eye(N), X1[tt, :]).T

        # training sample
        Y0 = Y[:, :T_training]
        X0 = X[:, :, :T_training]

        # effective sample
        YS = Y[:, T_training:]
        XS = X[:, :, T_training:]

        Y_obs = y[self.T0:, :]
        X_obs = X1[T_training:, :]

        return Y0, X0, YS, XS, Y_obs, X_obs


    def sur(self):
        """
        This function calculates a 2-step seemingly unrelated regression.
        This is used for initializing things (prior)

        Y    :  N x T matrix
                row indicating equations, column indicating time periods,
                i.e, column j is the observation of Y at time j.

        X    :  tensor object, with dim (K_1 + K_2 + ... + K_N) x N x T,
                where K_i is the number of independent variables for equation i,
                and T indicates time as above. For page t, we have (cap)X_t' as
                in equation (1.2) in Notes. So page t records observations of right
                hand side variables for N equations at time t.


        """
        Y0, X0 = self.create_YX()[:2]
        TT = self.T_training

        ry, cy = Y0.shape
        rx, cx, px = X0.shape


        #===================================
        # 1st stage estimates:
        #   ... weighting matrix = I
        #===================================
        Mxx = np.zeros((rx, rx))
        Mxy = np.zeros((rx,))

        for t in range(TT):
            Mxx = Mxx + X0[:, :, t] @ X0[:, :, t].T
            Mxy = Mxy + X0[:, :, t] @ Y0[:, t]

        theta = inv(Mxx) @ Mxy

        e = np.zeros((ry, TT))        # residuals
        for t in range(TT):
            e[:, t] = Y0[:, t] - X0[:, :, t].T @ theta

        #===================================
        # 2nd stage estimates:
        #   ... weighting matrix = inv(cov(e))
        #===================================
        W = inv(sp.cov(e))
        Mxx = np.zeros((rx, rx))
        Mxy = np.zeros((rx,))

        for t in range(TT):
            Mxx = Mxx + X0[:, :, t] @ W @ X0[:, :, t].T
            Mxy = Mxy + X0[:, :, t] @ W @ Y0[:, t]

        theta = inv(Mxx) @ Mxy

        for t in range(TT):
            e[:, t] = Y0[:, t] - X0[:, :, t].T  @ theta

        Vu = sp.cov(e)
        W = inv(Vu)

        Mxx = np.zeros((rx, rx))
        for t in range(TT):
            Mxx = Mxx + X0[:, :, t] @ W @ X0[:, :, t].T
        Vtheta = inv(Mxx)

        return theta, Vtheta, Vu


    def informative_prior(self):
        """
        Setting values for the hyperparameters
        """
        N, L = self.N, self.L
        numb_Xcols = 1 + N * L

        #--------------------------------------------------------------------------
        # (1) Informative prior form sur for theta mean and AR roots
        #--------------------------------------------------------------------------
        theta_bar, P_bar, RI = self.sur()

        df = N*numb_Xcols + 1       # prior degrees of freedom
        Q_bar = df*(3.5e-04)*P_bar  # prior covariance matrix for state innovations
        #TQ_bar = df*Q_bar;         # prior scaling matrix

        #--------------------------------------------------------------------------
        # (2) Priors for Stock Volatility parameters:
        #--------------------------------------------------------------------------
        # ... standard dev for volatility innovation ~ inverse gamma
        v0 = 1
        d0 = (.01)**2
        a0, b0 = np.ones(N)*v0, np.ones(N)*d0

        # ... correlation parameters for stochastic volatilities (normal)
        J = int(N * (N - 1) / 2)
        b_bar = np.zeros((J,))
        Pb_bar = 1e4 * np.eye(J)

        # ... log(h0) ~ normal (ballpark numbers)
        Ph_bar = 10 * np.eye(3)
        lnh_bar = np.log(np.diag(RI))

        return theta_bar, b_bar, Pb_bar, a0, b0, lnh_bar, Ph_bar, P_bar, Q_bar



def cs_model(Y_obs, X_obs, X1, theta_bar, b_bar, Pb_bar, a0, b0, lnh_bar, Ph_bar, P_bar, Q_bar, filename):
    """
    Inputs:
        - Y_obs: LHS variables from the measurement equation ( shape must be TxK )
        - X_obs: RHS variables from the measurement equation ( shape must be TxM )
        - theta_bar: prior mean for Theta_0 ( KM array )
        - P_bar: prior covariance for Theta_0 ( KMxKM array )
        - Q_bar: prior scale parameter for Q ( KMxKM array )

        - a0,b0: prior parameters for sigma^2 (K arrays)
        - lnh_bar: prior mean for ln(h_0)  (K array)
        - Ph_bar:  prior covariance for ln(h_0)  (K array)
        - b_bar: prior mean for the betas  (K(K-1)/2 array)
        - Pb_bar: prior covariance for the betas ( [K(K-1)/2]x[K(K-1)/2] array)

        - filename : (string) name of the storing file

    Outputs:
        - pymc model ready for sampling
    """

    T = Y_obs.shape[0]
    K = Y_obs.shape[1]
    M = X_obs.shape[1]
    KM = theta_bar.size
    J = int(K * (K - 1) / 2)

    #---------------------------------------------------
    # Define the priors for Q^{-1}
    #---------------------------------------------------
    Q_inv = pm.Wishart("Q_inv", n = K * M + 1, Tau = Q_bar)

    #---------------------------------------------------
    # Define Theta as a list containing the elements of theta^T
    #---------------------------------------------------
    Theta = [pm.MvNormalCov('Theta_0', theta_bar, P_bar)]
    for i in range(1, T + 1):
        Theta.append(pm.MvNormal('Theta_%d' % i, Theta[i - 1], Q_inv))

    #---------------------------------------------------
    # Define Sigma2 as a list containing all sigma^2_{i}
    #---------------------------------------------------
    Sigma2 = [pm.InverseGamma('sigma2_1', a0[0], b0[0])]
    for i in range(1, K):
        Sigma2.append(pm.InverseGamma('sigma2_%d' % (i + 1), a0[i], b0[i]))

    #---------------------------------------------------
    # Define LH as a list containing all ln(h_{i,t}) -- this is log( H^T )
    #---------------------------------------------------

    # Use a deterministic variable for the covariance matrix of LH
    Cov_lnH = pm.Lambda('Cov_lnH', lambda s = Sigma2: np.diag(s) )

    LH = [pm.MvNormalCov('lnh_0', lnh_bar, Ph_bar)]
    for i in range(1, T + 1):
        LH.append(pm.MvNormalCov('lnh_%d' % i, LH[i - 1], Cov_lnH))

    #---------------------------------------------------
    # Define Betas: if K=1 (only one observable), there is no covariance
    #---------------------------------------------------
    if J > 0:
        Betas = pm.MvNormalCov('betas', b_bar, Pb_bar)
    else:
        # by making it observed we fix the value of this stochastic variable
        Betas = pm.MvNormalCov('betas', b_bar, Pb_bar, value = np.asarray([]), observed = True)

    #---------------------------------------------------
    # Y's are observed, but we have to define them as stochastic variables and set observed to True
    #---------------------------------------------------

    # Use deterministic variables for R_t and collect them in an ordered list
    Binv = pm.Lambda('Binv', lambda b = Betas: inv(B_tril(b)))
    R = [pm.Lambda('R_%d' % 1, lambda b = Binv, lh = LH[1]: b @ np.diag(np.exp(lh)) @ b.T)]

    # Use deterministic variables for conditional means of Y_t -> list muY containing all
    muY = [pm.Lambda('muY_%d' % 1, lambda yy = X_obs[0, :], th = Theta[1]: yy @ th.reshape(K, M).T)]

    y = [pm.MvNormalCov("Y_1", muY[0], R[0], value = Y_obs[0,:], observed = True)]
    for i in range(1, T):
        muY.append(pm.Lambda('muY_%d' % (i + 1), lambda yy = X_obs[i, :], th = Theta[i + 1]: yy @ th.reshape(K, M).T))
        R.append(pm.Lambda('R_%d' % (i + 1), lambda b = Binv, lh = LH[i + 1]: b @ np.diag(np.exp(lh)) @ b.T ))
        y.append(pm.MvNormalCov("Y_%d" % (i + 1), muY[i], R[i], value = Y_obs[i,:], observed = True))

    # Need to convert the lists to pymc Container arrays
    Theta, muY, y, LH, R = pm.Container(Theta), pm.Container(muY), pm.Container(y), pm.Container(LH), pm.Container(R)
    Sigma2 = pm.Container(Sigma2)

    m = pm.Model([Theta, y, muY, Q_inv, LH, Sigma2, Betas, R, Cov_lnH])
    mcmc = pm.MCMC(m, db = 'pickle', dbname = '../data/posterior_pymc/' + str(filename))

    # Assign the step methods to the unobserved stochastic variables
    mcmc.use_step_method(ForwardBackward, Theta, y, R, Q_inv, X1)
    mcmc.use_step_method(W_Q, Q_inv, Theta)
    mcmc.use_step_method(IG_Sigma, Sigma2, LH)
    mcmc.use_step_method(N_Beta, Betas, y, muY, LH)
    mcmc.use_step_method(Metropolis_LH, LH, y, muY, Betas, Sigma2)

    return mcmc, Theta, LH, Q_inv, Sigma2, Betas, R

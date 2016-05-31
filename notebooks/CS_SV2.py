from __future__ import division
import pymc as pm
import numpy as np
import scipy as sp
from scipy.linalg import inv
from math import ceil


def B_tril(b):
    """_
    This function builds a lower triangular matrix from the elements of b (array), so 
    that the diagonal elements are all 1s (corresponds to the B matrix in the paper).
    """
    bb, place = b.tolist(), 0
    K = int(ceil(np.sqrt(len(bb)*2)))
    if K>0:                 # if b contains any element
        bb.insert(place, 1.)
        for ii in range(1,K):
            place += ii+1
            bb.insert(place,1.)
        B, indices = np.eye(K), np.tril_indices(K)
        B[indices] = bb
        return B
    else:                   # if b is empty it gives back B=1
        return np.asarray([[1]])

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
    
    def __init__(self, stochastic, Y, R, Q_inv, X_obs, verbose=None):
        # Since StepMethod stores Theta as an unordered set in self.stochastics, we need 
        # to use a trick to save the ordering of Theta_i in self.my_list 
        x_dict = {x:y for y, x in enumerate(stochastic)}
        # Since we subclass StepMethod directly, the result of the line below is to create
        # self.stochastics (notice the plural form!) which is a set of stochastic variables
        pm.StepMethod.__init__(self, stochastic, verbose)
        my_list = [None]*len(self.stochastics)
        for element in self.stochastics:
            my_list[x_dict[element]] = element
        self.my_list = my_list

        # Data (observed stochastic variables)
        self.Y, self.X = Y, X_obs

        # Size of numpy arrays        
        self.T = np.asarray(self.Y.value).shape[0]
        self.K = np.asarray(self.Y.value).shape[1]                  # number of equations in the VAR
        self.M = X_obs.shape[1]                                     # M = (1 + K*L)
        self.KM = self.K*self.M                                      
        
        # Other non-observed stochastic variables
        self.Qinv = Q_inv
        self.R = R
    
        # Initial moments for the prior        
        self.theta0 = self.my_list[0].parents['mu']
        self.P0 = self.my_list[0].parents['C']

        # Auxiliary matrix for the rejection sampling (to construct a test matrix A)       
        self.L = (self.M-1)/self.K                                  
        self.S = np.hstack((np.eye(self.K*(self.L-1)), np.zeros((self.K*(self.L-1),self.K))))

    @staticmethod
    def competence(stochastics):
        """
        We allow to apply this step_method only manually via MCMC.use_step_method().
        """
        return 0        
    
    def propose(self):
        Q = sp.linalg.solve( self.Qinv.value, np.eye(self.KM))
        R = self.R.value
        y = np.asarray(self.Y.value, dtype=float)

        P = np.empty((self.T+1, self.KM, self.KM), dtype=float)                # P_{t-1|t-1}
        P_ = np.empty((self.T, self.KM, self.KM), dtype=float)                 # P_{t|t-1}
        Th = np.empty((self.KM, self.T+1), dtype=float)                        # Theta_{t-1|t-1}
        theta = np.empty((self.KM, self.T+1), dtype=float)                     # new draw 
        
        XX = np.empty((self.K, self.KM), dtype=float)
        K = np.empty((self.KM, self.K), dtype=float)
    
        P[0,:,:] = self.P0                                                    
        Th[:,0] = self.theta0                                               

        #-----------------------------------------------------
        # FORWARD STEP (Kalman filter):
        #-----------------------------------------------------
        for tt in range(self.T):
            XX[:,:] = np.kron(np.eye(self.K), self.X[tt,:])
            P_[tt,:,:] = P[tt,:,:] + Q    
            K[:,:] = sp.linalg.solve(np.dot(np.dot(XX, P_[tt,:,:]), XX.T) + R[tt], np.dot(XX, P_[tt,:,:])).T

            Th[:,tt+1] = Th[:,tt] + np.dot(K, y[tt,:].T - np.dot(XX, Th[:,tt]) )
            P[tt+1,:,:] = P_[tt,:,:] - np.dot( np.dot(K,XX), P_[tt,:,:] )
        
        #-----------------------------------------------------
        # BACKWARD STEP:
        #-----------------------------------------------------
        theta[:,-1] = pm.rmv_normal_cov(Th[:,-1].ravel(), P[-1,:,:])
        
        for tt in range(2, self.T+2):
            theta[:,-tt] = pm.rmv_normal_cov((Th[:,-tt] + np.dot(P[-tt,:,:],sp.linalg.solve( P_[-(tt-1),:,:], theta[:,-(tt-1)]-Th[:,-tt]) )).ravel(),\
                                                P[-tt,:,:] - np.dot(P[-tt,:,:], sp.linalg.solve( P_[-(tt-1),:,:], P[-tt,:,:])))         
        # Assigning new values
        # remember: my_list is a sorted list starting with Theta_0, Theta_1, etc..
        for ind, stoch in enumerate(self.my_list):
            stoch.value = theta[:, ind]

    #-----------------------------------------------------
    # REJECTION SAMPLING: 
    #-----------------------------------------------------
    def step(self):
        accept = 0
        
        while accept==1:
            # Draw a proposal   
            self.propose()
            lmax = np.empty(self.T+1)
            ind=0
            # Reject the proposal if any element of Theta violates the stability condition        
            for ss in self.stochastics:
                theta_test = ss.value
                A = [theta_test[1:self.M].tolist()]
                for i in range(1,self.K):
                    A.append( theta_test[1+i*self.M:(i+1)*self.M].tolist() )        
                A = np.vstack((np.asarray(A),self.S))
                lmax[ind] = abs(sp.linalg.eig(A)[0]).max()
                ind += 1
            if lmax.max()>1:
                accept = 0
            else:
                accept = 1



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

    def __init__(self, stochastic, Theta, verbose=None):
        pm.Gibbs.__init__(self, stochastic, verbose)
        self.conjugate = True   # pymc will include a Metropolis rejection step if this is false

        # Other non-observed stochastic variables
        self.Theta = Theta
 
        # Initial moments for the prior
        self.df = stochastic.parents["n"]
        self.Q_bar = stochastic.parents["Tau"]

    
    def propose(self):
        theta = np.asarray(self.Theta.value)
        v = theta[1:]-theta[:-1]
        TT = v.shape[0]

        self.stochastic.value = pm.rwishart(self.df + TT, self.Q_bar + np.dot(v.T,v))




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
    

    def __init__(self, stochastic, LH, verbose=None):
        # Since StepMethod stores Theta as an unordered set in self.stochastics, we need 
        # to use a trick to save the ordering of Theta_i in self.my_list 
        x_dict = {x:y for y, x in enumerate(stochastic)}
        pm.StepMethod.__init__(self, stochastic, verbose)
        my_list = [None]*len(self.stochastics)
        for element in self.stochastics:
            my_list[x_dict[element]] = element
        self.my_list = my_list

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
    
    @staticmethod
    def competence(stochastics):
        return 0        

    def step(self):
        dlnh = np.asarray(self.LH.value)[1:,:] - np.asarray(self.LH.value)[:-1,:]
        TT = dlnh.shape[0]
        a1 = self.a0 + TT/2
        b1 = self.b0 + np.sum(dlnh**2,0)/2
        
        for ind, stoch in enumerate(self.my_list):
            stoch.value = pm.rinverse_gamma(a1[ind], b1[ind])





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

    def __init__(self, stochastic, Y, mu_Y, LH, verbose=None):
        pm.Gibbs.__init__(self, stochastic, verbose)
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
        b_new = np.empty_like( self.stochastic.value )

        # auxiliary variables to pick the right subvector/submatrix for the equations 
        lb = 0  
        ub = 1

        for j in range(1,K):
            z = np.expand_dims(H[:,j],1)*np.expand_dims(ee[:,j],1)      # LHS variable in the regression
            Z = np.expand_dims(-H[:,j],1)*ee[:,:j]                      # RHS variables in the regression
            
            b_prior = np.asarray([self.b_bar[lb:ub]])
            Vinv_prior = inv(self.Pb_bar[lb:ub, lb:ub])

            V_post = inv( Vinv_prior + np.dot(Z.T, Z) )
            b_post = V_post.dot( np.dot(Vinv_prior, b_prior.T) + np.dot(Z.T, z) )

            b_new[lb:ub] = pm.rmv_normal_cov( b_post.ravel(), V_post )
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

    def __init__(self, stochastic, Y, mu_Y, Betas, Sigma2, verbose=None):
        # Since StepMethod stores Theta as an unordered set in self.stochastics, we need 
        # to use a trick to save the ordering of Theta_i in self.my_list 
        x_dict = {x:y for y, x in enumerate(stochastic)}
        pm.StepMethod.__init__(self, stochastic, verbose)
        my_list = [None]*len(self.stochastics)
        for element in self.stochastics:
            my_list[x_dict[element]] = element
        self.my_list = pm.Container( my_list )

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

    @staticmethod
    def competence(stochastics):
        return 0        
    
    def step(self):
        sig2  = np.asarray(self.Sigma2.value)
        ee = np.asarray(self.Y.value) - np.asarray(self.muY.value)
        BB = B_tril(self.Betas.value)
        ut = np.dot(BB, ee.T).T
        lnH_old = np.asarray(self.my_list.value)
        lnH_new = np.empty_like(lnH_old)

        #--------------------------------------
        # update the initial log(h_0)
        #--------------------------------------
        ss0 = np.diag(self.Ph_bar)
        ss = ss0*sig2/(sig2 + ss0)
        lnh0_cov = np.diag(ss)
        lnh0_mu = ss*(self.lnh_bar/ss0 + lnH_old[1,:]/sig2 )
        lnH_new[0,:] = pm.rmv_normal_cov( lnh0_mu.ravel(), lnh0_cov )

        #--------------------------------------
        # update log(h_t) step-by-step for 0<t<T
        #--------------------------------------
        for tt in range(1,self.T):
            # mean and variance for log(h) (proposal density)
            lnh_mu = .5*( lnH_new[tt-1,:] + lnH_old[tt+1,:] )
            lnh_cov = np.diag( .5*sig2 )

            # candidate draw from normal
            lnh_prop = pm.rmv_normal_cov( lnh_mu.ravel(), lnh_cov )

            # acceptance probability
            lp1 = -0.5*lnh_prop - ut[tt-1,:]**2/(2*np.exp(lnh_prop))
            lp0 = -0.5*lnH_old[tt,:] - ut[tt-1,:]**2/(2*np.exp(lnH_old[tt,:]))
            accept = np.minimum(1., np.exp(lp1 - lp0))

            u = np.random.rand(self.K)
            for kk in range(self.K):
                if u[kk] <= accept[kk]:
                    lnH_new[tt,kk] = lnh_prop[kk]
                else:
                    lnH_new[tt,kk] = lnH_old[tt,kk]

        #--------------------------------------
        # update the last h_T
        #--------------------------------------
        # mean and variance for log(h) (proposal density)
        lnh_mu = lnH_new[self.T-1,:]
        lnh_cov = np.diag(sig2)

        # candidate draw from normal
        lnh_prop = pm.rmv_normal_cov( lnh_mu.ravel(), lnh_cov )

        # acceptance probability
        lp1 = -0.5*lnh_prop - ut[-1,:]**2/(2*np.exp(lnh_prop))
        lp0 = -0.5*lnH_old[self.T,:] - ut[-1,:]**2/(2*np.exp(lnH_old[self.T,:]))
        accept = np.minimum(1., np.exp(lp1 - lp0))

        u = np.random.rand(self.K)
        for kk in range(self.K):
            if u[kk] <= accept[kk]:
                lnH_new[self.T,kk] = lnh_prop[kk]
            else:
                lnH_new[self.T,kk] = lnH_old[self.T,kk]


        #--------------------------------------
        # Assigning new values
        #--------------------------------------
        for ind, stoch in enumerate(self.my_list):
            stoch.value = lnH_new[ind,:]













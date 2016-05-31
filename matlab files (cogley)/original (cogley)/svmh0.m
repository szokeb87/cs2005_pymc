function [h,mu,ss] = svmh0(hlead,alpha,delta,sv,mu0,ss0);

% h = svmh0(hlead,alpha,delta,sv,mu0,ss0);

% This file returns a draw from the posterior conditional density
% for the stochastic volatility parameter at time 0.  This is conditional
% on the first period realization, hlead, as well as the prior and parameters 
% of the svol process.  

% mu0 and ss0 are the prior mean and variance.  The formulas simplify if these are
% given by the unconditional mean and variance implied by the state, but we haven't
% imposed this.  (allows for alpha = 0, delta = 1) 

% Following JPR (1994), we use a MH step, but with a simpler log-normal proposal density. 
% (Their proposal is coded in jpr.m.) 

% mean and variance for log(h) (proposal density)
ssv = sv^2;
ss = ss0*ssv/(ssv + (delta^2)*ss0);
mu = ss*(mu0/ss0 + delta*(log(hlead) - alpha)/ssv);

% draw from lognormal (accept = 1, since there is no observation)
h = exp(mu + (ss^.5)*randn(1,1));


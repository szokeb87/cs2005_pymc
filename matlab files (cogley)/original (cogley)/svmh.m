function [h,R] = svmh(hlead,hlag,alpha,delta,sv,yt,hlast);

% h = svmh(hlead,hlag,alpha,delta,sv,y,hlast);

% This file returns a draw from the posterior conditional density
% for the stochastic volatility parameter at time t.  This is conditional
% on adjacent realizations, hlead and hlag, as well as the data and parameters 
% of the svol process.  

% hlast is the previous draw in the chain, and is used in the acceptance step. 
% R is a dummy variable that takes a value of 1 if the trial is rejected, 0 if accepted.

% Following JPR (1994), we use a MH step, but with a simpler log-normal proposal density. 
% (Their proposal is coded in jpr.m.) 

% mean and variance for log(h) (proposal density)
mu = alpha*(1-delta) + delta*(log(hlead)+log(hlag))/(1+delta^2);
ss = (sv^2)/(1+delta^2);

% candidate draw from lognormal
htrial = exp(mu + (ss^.5)*randn(1,1));

% acceptance probability
lp1 = -0.5*log(htrial) - (yt^2)/(2*htrial);
lp0 = -0.5*log(hlast) - (yt^2)/(2*hlast);
accept = min(1,exp(lp1 - lp0));

u = rand(1);
if u <= accept,
   h = htrial;
   R = 0;
else
   h = hlast;
   R = 1;
end

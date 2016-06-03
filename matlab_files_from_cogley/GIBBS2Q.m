function QA = GIBBS2Q(TQ,df,N,L);

% function QA = GIBBS2Q(TQ,df,N,L);

% This file executes the second stage of the Gibbs
% sampler, generating random draws for Q 
% using the inverse-wishart density. The scale matrix 
% is T times the usual covariance estimator, and
% E(QA) is approximately (1/T)QR

% df = degrees of freedom
% inv(Q) is the scale matrix for 
% Q
% TQ is N(1+NL) x N(1+NL)

PS = real(sqrtm(inv(TQ))); 
u = randn(N*(1+N*L),df);
QA = inv(PS*u*u'*PS');

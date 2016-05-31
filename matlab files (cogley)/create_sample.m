NG = 5000; % number of draws from Gibbs sampler per data file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data, partition sample
load('NEWQDATA.MAT')

%  dependent variables
y = [y3,.01*log(ur./(1-ur)),dp];
clear dp ur y3 newqdata date
[T,N] = size(y); % N = number of equations
L = 2; % VAR lag order

% initial estimates through T0
T0 = 4*11-L-1; %end of 1958
% adaptive estimation through T1
T1 = T-L; % end of 2000

% lagged dependent variables: X1 = [y(t-1) y(t-1) ... y(t-L)]
X1 = zeros(T-L,1+(N*L));
X1(1:T-L,1) = ones(T-L,1);
for i = 1:L,
   X1(1:T-L,2+(N*(i-1)):1+(i*N)) = y(1+L-i:T-i,:);
end
Y = y(1+L:T,:)';
X = xmat(X1',X1',X1'); % X is a tensor; identical rhs variables in this case

% partitioning the data
Y0 = Y(:,1:T0);
YS = Y(:,1+T0:T1);
X0 = X(:,:,1:T0);
XS = X(:,:,1+T0:T1);
X01 = X1(1:T0,:);
XS1 = X1(1+T0:T1,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% an informative prior: SUR estimates through T0
[SI,PI,RI] = sur(Y0,X0,T0);
df = N*(1+N*L) + 1; % prior degrees of freedom
DF = df + T1 - T0; % posterior degrees of freedom
Q0 = (3.5e-04)*PI; % prior covariance matrix for state innovations
TQ0 = df*Q0; % prior scaling matrix
TQ = TQ0; % initialize posterior scaling matrix

% implied prior mean and AR roots
S = [eye(N*(L-1)),zeros(N*(L-1),N)];
A = [SI(2:1+N*L,1)'; SI(2+(1+N*L):2*(1+N*L),1)'; SI(2+2*(1+N*L):3*(1+N*L),1)';S];
rr = eig(A);
M = inv(eye(N*L) - A)*[SI(1,1); SI(2+N*L,1); SI(3+2*N*L,1); zeros(N*(L-1),1)];
mm = M(1:3,1);
'prior mean'; mm
'prior AR roots'; rr
clear M mm rr

% clear initial sample
clear y Y X Y0 X0 X01 T0 T1
[T,N] = size(YS');

% priors for SVOL parameters
% alpha, delta (normal) (same for all stochastic volatilities) (not needed for random walk spec)
b0 = [0,1]';
k = size(b0,1);
Sb0 = 10000*eye(k,k);
b = zeros(2,N); % alpha, delta in log volatility regression

% priors for sv (inverse gamma) (standard dev for volatility innovation)
sv0 = .01;
v0 = 1;
d0 = sv0^2;
eh = zeros(T,N); % volatility innovations

% priors for log h0 (normal); ballpark numbers
ss0 = 10;
mu0 = log(diag(RI));

% prior for correlation parameters for stochastic volatilities (normal)
ch0 = zeros(N*(N-1)/2,1);
ch = ch0;
ssc0 = 10000*eye(N*(N-1)/2,N*(N-1)/2);
CF = chofac(N,ch0);
NW = NG/10;


%%

THETA_sample = zeros(N*(1+N*L),T,NG);                % theta
Q_sample = zeros(N*(1+N*L),N*(1+N*L),NG);            % Q
SIG_sample = zeros(N,NG);                            % sigma
H_sample = zeros(T+1,N,NG);                          % h
BET_sample = zeros(N*(N-1)/2,NG);                    % beta

load svol11
THETA_sample(:,:,1:NW) = SD;
Q_sample(:,:,1:NW) = QD;
SIG_sample(:,1:NW) = VD;
H_sample(:,:,1:NW) = HD;
BET_sample(:,1:NW) = CD;

load svol12
lb = 2; 
THETA_sample(:,:,((lb-1)*NW+1):lb*NW ) = SD;
Q_sample(:,:,((lb-1)*NW+1):lb*NW ) = QD;
SIG_sample(:,((lb-1)*NW+1):lb*NW ) = VD;
H_sample(:,:,((lb-1)*NW+1):lb*NW ) = HD;
BET_sample(:,((lb-1)*NW+1):lb*NW ) = CD;

load svol13
lb = 3; 
THETA_sample(:,:,((lb-1)*NW+1):lb*NW ) = SD;
Q_sample(:,:,((lb-1)*NW+1):lb*NW ) = QD;
SIG_sample(:,((lb-1)*NW+1):lb*NW ) = VD;
H_sample(:,:,((lb-1)*NW+1):lb*NW ) = HD;
BET_sample(:,((lb-1)*NW+1):lb*NW ) = CD;


load svol14
lb = 4; 
THETA_sample(:,:,((lb-1)*NW+1):lb*NW ) = SD;
Q_sample(:,:,((lb-1)*NW+1):lb*NW ) = QD;
SIG_sample(:,((lb-1)*NW+1):lb*NW ) = VD;
H_sample(:,:,((lb-1)*NW+1):lb*NW ) = HD;
BET_sample(:,((lb-1)*NW+1):lb*NW ) = CD;

load svol15
lb = 5; 
THETA_sample(:,:,((lb-1)*NW+1):lb*NW ) = SD;
Q_sample(:,:,((lb-1)*NW+1):lb*NW ) = QD;
SIG_sample(:,((lb-1)*NW+1):lb*NW ) = VD;
H_sample(:,:,((lb-1)*NW+1):lb*NW ) = HD;
BET_sample(:,((lb-1)*NW+1):lb*NW ) = CD;

load svol16
lb = 6; 
THETA_sample(:,:,((lb-1)*NW+1):lb*NW ) = SD;
Q_sample(:,:,((lb-1)*NW+1):lb*NW ) = QD;
SIG_sample(:,((lb-1)*NW+1):lb*NW ) = VD;
H_sample(:,:,((lb-1)*NW+1):lb*NW ) = HD;
BET_sample(:,((lb-1)*NW+1):lb*NW ) = CD;

load svol17
lb = 7; 
THETA_sample(:,:,((lb-1)*NW+1):lb*NW ) = SD;
Q_sample(:,:,((lb-1)*NW+1):lb*NW ) = QD;
SIG_sample(:,((lb-1)*NW+1):lb*NW ) = VD;
H_sample(:,:,((lb-1)*NW+1):lb*NW ) = HD;
BET_sample(:,((lb-1)*NW+1):lb*NW ) = CD;

load svol18
lb = 8; 
THETA_sample(:,:,((lb-1)*NW+1):lb*NW ) = SD;
Q_sample(:,:,((lb-1)*NW+1):lb*NW ) = QD;
SIG_sample(:,((lb-1)*NW+1):lb*NW ) = VD;
H_sample(:,:,((lb-1)*NW+1):lb*NW ) = HD;
BET_sample(:,((lb-1)*NW+1):lb*NW ) = CD;

load svol19
lb = 9; 
THETA_sample(:,:,((lb-1)*NW+1):lb*NW ) = SD;
Q_sample(:,:,((lb-1)*NW+1):lb*NW ) = QD;
SIG_sample(:,((lb-1)*NW+1):lb*NW ) = VD;
H_sample(:,:,((lb-1)*NW+1):lb*NW ) = HD;
BET_sample(:,((lb-1)*NW+1):lb*NW ) = CD;

load svol20
lb = 10; 
THETA_sample(:,:,((lb-1)*NW+1):lb*NW ) = SD;
Q_sample(:,:,((lb-1)*NW+1):lb*NW ) = QD;
SIG_sample(:,((lb-1)*NW+1):lb*NW ) = VD;
H_sample(:,:,((lb-1)*NW+1):lb*NW ) = HD;
BET_sample(:,((lb-1)*NW+1):lb*NW ) = CD;


save('sample.mat','THETA_sample','Q_sample','SIG_sample','H_sample','BET_sample');
tr_Q = zeros(NG,1);

for i=1:NG
    tr_Q(i,1)=trace(Q_sample(:,:,i));
end

figure(1);
histogram(tr_Q); hold on;
line([trace(Q0), trace(Q0)],[0,500]); hold off;
title('tr(Qbar) and the posterior distribution for tr(Q)')
    
theta_mean = squeeze(mean(THETA_sample,3));
theta_std = squeeze(std(THETA_sample,0,3));

figure(2)
subplot(1,2,1)
plot(theta_mean')
title('Posterior mean of Theta')
subplot(1,2,2)
plot(theta_std')
title('Posterior stdev of Theta')

H_mean = zeros(5000,T+1,3,3);

for ss=1:5000
    BI = inv(chofac(3,BET_sample(:,ss)));
    for tt=1:(T+1)
        SW = diag(H_sample(tt,:,ss));
        R = BI*SW*BI';
        H_mean(ss,tt,:,:) = R;
    end
end

HH_mean = squeeze(mean(H_mean,1));
nomint = sqrt(HH_mean(:,1,1));
unemp = sqrt(HH_mean(:,2,2));
inflation = sqrt(HH_mean(:,3,3));

corr_nu = HH_mean(:,1,2)./( sqrt(HH_mean(:,1,1)).*sqrt(HH_mean(:,2,2)));
corr_ni = HH_mean(:,1,3)./( sqrt(HH_mean(:,1,1)).*sqrt(HH_mean(:,3,3)));
corr_iu = HH_mean(:,2,3)./( sqrt(HH_mean(:,2,2)).*sqrt(HH_mean(:,3,3)));

ldR = zeros(T+1,1);
for tt= 1:(T+1)
    ldR(tt,1) = log(det(squeeze(HH_mean(tt,:,:))));
end

figure(3)
subplot(3,2,1)
plot(10000*nomint)
title('Standard error (interest rate)')
axis([0, T+1, 0, 70 ])

subplot(3,2,3)
plot(10000*inflation)
title('Standard error (inflation)')
axis([0, T+1, 25, 70 ])
subplot(3,2,5)
plot(10000*unemp)
title('Standard error (unemployment)')
axis([0, T+1, 2, 7 ])

subplot(3,2,2)
plot(corr_nu)
title('Corr (int rate vs unemployment)')
axis([0, T+1, -0.3, 0 ])
subplot(3,2,4)
plot(corr_ni)
title('Corr (int rate vs inflation)')
axis([0, T+1, 0.1, 0.9 ])
subplot(3,2,6)
plot(corr_iu)
title('Corr (inflation vs unemployment)')
axis([0, T+1, -0.3, 0.02 ])

figure(4)
plot(ldR)
title('Total prediction variance, log(det(E R_t))')
axis([0, T+1, -43, -36 ])

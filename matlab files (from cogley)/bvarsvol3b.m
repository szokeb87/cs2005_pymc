%  This simulates a trivariate var with stochastic volatility.

%  This version warms up the svol with 1000 draws from the distribution
%  for h,B,sv given dy.  Ie, it shuts down the VAR parameters until
%  the svol gets going.

%  Version 3a had convergence problems


NG = 5000; % number of draws from Gibbs sampler per data file
NF = 20;   % number of data files (memory constraint)
D = 10;    % sampling interval from markov chain (memory constraint, mixing)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% files and variable names

% catalog of data files
DFILE(1,:) = ['svol01'];
DFILE(2,:) = ['svol02'];
DFILE(3,:) = ['svol03'];
DFILE(4,:) = ['svol04'];
DFILE(5,:) = ['svol05'];
DFILE(6,:) = ['svol06'];
DFILE(7,:) = ['svol07'];
DFILE(8,:) = ['svol08'];
DFILE(9,:) = ['svol09'];
DFILE(10,:) = ['svol10'];
DFILE(11,:) = ['svol11'];
DFILE(12,:) = ['svol12'];
DFILE(13,:) = ['svol13'];
DFILE(14,:) = ['svol14'];
DFILE(15,:) = ['svol15'];
DFILE(16,:) = ['svol16'];
DFILE(17,:) = ['svol17'];
DFILE(18,:) = ['svol18'];
DFILE(19,:) = ['svol19'];
DFILE(20,:) = ['svol20'];

% variables to be saved
varname(1,:) = ['SD']; % states
varname(2,:) = ['QD']; % state innovation variance
varname(3,:) = ['HD']; % stochastic volatilities
varname(4,:) = ['CD']; % cholesky factor
varname(5,:) = ['VD']; % standard deviation of vol. innovation
%varname(6,:) = ['RD']; % regression parameters (not used for random walk)

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
CF = chofac(N, ch0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize gibbs arrays

SA = zeros(N*(1+N*L),T,NG);                 % draws of the state vector
QA = zeros(N*(1+N*L),N*(1+N*L),NG);         % draws of covariance matrix for state innovations
%RP = zeros(2,N,NG); % alpha, delta regression parameters (not needed for random walk)
SV = zeros(N,NG);                           % standard error for volatility innovation
H = zeros(T+1,N,NG);                        % stochastic volatilities
CH = zeros(N*(N-1)/2,NG);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initial for log(h(t))

lh(1:2,:) = ones(2,1)*mu0';
dy = diff(YS');
e = dy - ones(T-1,1)*mean(dy);
lh(3:T+1,:) = log(e.^2);
H0 = exp(lh);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% warm up for stochastic volatilities; shut down var, treat dy as var innovation

evar = [zeros(3,1) e'];
H(:,:,1) = H0;
for iter = 2:NG,
    % R conditional on states and data (svol programs)
    lh = log(H(:,:,iter-1));
    for i = 1:N,
       % sv|b,lh,ch,y
       eh(:,i) = lh(2:T+1,i) - lh(1:T,i);  % random walk
       v = IG2(v0, d0, eh(:,i));
       SV(i,iter) = v^.5;
     end

     % orthogonalize var innovations
     CF = chofac(N, ch);
     f = (CF*evar)';

     % lh|ch,sv,b1,y
     for i = 1:N,
       H(1,i,iter) = svmh0(H(2, i, iter-1), 0, 1, SV(i, iter), mu0(i, 1), ss0);
       for t = 2:T,
          H(t,i,iter) = svmh(H(t+1, i, iter-1), H(t-1, i, iter), 0, 1, SV(i, iter), f(t-1, i), H(t, i, iter-1));
       end
       H(T+1, i, iter) = svmhT(H(T, i, iter), 0, 1, SV(i, iter), f(T, i), H(T+1, i, iter-1));
     end

     % ch|sv,b,lh,y
     k = 0;
     for i = 2:N,
        lhs = H(2:T+1,i,iter).^.5;
        for n = 1:N,
           yhs(:,n) = (evar(n,:)')./lhs;
        end
        yr = yhs(:,i);
        xr = -yhs(:,1:i-1);
        j = k+1;
        k = i-1+k;
        ch(j:k,1) = bayesreg(ch0(j:k), ssc0(j:k,j:k), 1, yr,xr);
     end
     CH(:,iter) = ch;
     iter
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initial draw of states

H0 = H(:, :, NG);
[S0, P0, P1] = KFR(YS, XS, Q0, CF, H0, SI, PI, T, N, L);
mlmax = 2;
while mlmax >= 1,
     SA1 = GIBBS1(YS,XS,S0,P0,P1,T,N,L);
     for j = 1:T,
        A = [SA1(2:1 + N*L, j)';
             SA1(2 + (1 + N*L):2*(1 + N*L), j)';
             SA1(2 + 2*(1 + N*L):3*(1 + N*L), j)';
             S];
        lmax(j, 1) = max(abs(eig(A)))';
     end
     mlmax = max(lmax)
end

for i = 1:4,
    SA(:,:,i) = SA1;
end
H(:,:,1:4) = H(:,:,NG-3:NG);
CH(:,1:4) = CH(:,NG-3:NG);
SV(:,1:4) = SV(:,NG-3:NG);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% begin MCMC

for file = 1:NF,
maxshakes = 50; % maximum number of attempts at stable draw (not needed for this sample)
iter = 5; % provide buffer for back steps in subsequent files
gc = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while iter <= NG,

    % Draw Q conditional on states and data
    [TQ,DF] = IWPQ(SA(:,:,iter-1),N,T,L,TQ0,df);
    QA(:,:,iter) = GIBBS2Q(TQ,DF,N,L);

    % R conditional on states and data (svol programs)
    lh = log(H(:,:,iter-1));
    for i = 1:N,
       % b|lh,ch,sv,y (not needed for random walk)
       %yr = lh(2:T+1,i);
       %xr = [ones(size(yr)),lh(1:T,i)];
       %RP(:,i,iter) = bayesreg(b0,Sb0,SV(i,iter-1),yr,xr); % alpha, delta

       % sv|b,lh,ch,y
       %eh(:,i) = lh(2:T+1,i) - RP(1,i,iter)*ones(T,1) - RP(2,i,iter)*lh(1:T,i);  %AR1
       eh(:,i) = lh(2:T+1,i) - lh(1:T,i);  % random walk
       v = IG2(v0,d0,eh(:,i));
       SV(i,iter) = v^.5;
     end

     % orthogonalize var innovations
     evar = innovm(YS, XS1, SA(:,:,iter-1), N, T, L);
     CF = chofac(N,ch);
     f = (CF*evar)';

     % lh|ch,sv,b1,y
     for i = 1:N,
       %H(1,i,iter) = svmh0(H(2,i,iter-1),RP(1,i,iter),RP(2,i,iter),SV(i,iter),mu0(i,1),ss0);
       H(1,i,iter) = svmh0(H(2,i,iter-1),0,1,SV(i,iter),mu0(i,1),ss0);
       for t = 2:T,
          %H(t,i,iter) = svmh(H(t+1,i,iter-1),H(t-1,i,iter),RP(1,i,iter),RP(2,i,iter),SV(i,iter),f(t-1,i),H(t,i,iter-1));
          H(t,i,iter) = svmh(H(t+1,i,iter-1),H(t-1,i,iter),0,1,SV(i,iter),f(t-1,i),H(t,i,iter-1));
       end
       %H(T+1,i,iter) = svmhT(H(T,i,iter),RP(1,i,iter),RP(2,i,iter),SV(i,iter),f(T,i),H(T+1,i,iter-1));
       H(T+1,i,iter) = svmhT(H(T,i,iter),0,1,SV(i,iter),f(T,i),H(T+1,i,iter-1));
     end

     % ch|sv,b,lh,y
     k = 0;
     for i = 2:N,
        lhs = H(2:T+1,i,iter).^.5;
        for n = 1:N,
           yhs(:,n) = (evar(n,:)')./lhs;
        end
        yr = yhs(:,i);
        xr = -yhs(:,1:i-1);
        j = k+1;
        k = i-1+k;
        ch(j:k,1) = bayesreg(ch0(j:k),ssc0(j:k,j:k),1,yr,xr);
     end
     CH(:,iter) = ch;

     % states conditional on hyperparameters and data
     [S0,P0,P1] = KFR(YS,XS,QA(:,:,iter),CF,H(:,:,iter),SI,PI,T,N,L);
     SA(:,:,iter) = GIBBS1(YS,XS,S0,P0,P1,T,N,L);

     % check stability; reject unstable draws
     for j = 1:T,
        A = [SA(2:1+N*L,j,iter)'; SA(2+(1+N*L):2*(1+N*L),j,iter)'; SA(2+2*(1+N*L):3*(1+N*L),j,iter)';S];
        lmax(j,1) = max(abs(eig(A)))';
     end

     if max(lmax) <= 1,
        [file,iter]
        max(eig(QA(:,:,iter)))
        iter = iter+1;
        gc = 1;
     elseif (max(lmax) > 1)&(gc < maxshakes),
        'States unstable, try again'
        gc = gc + 1
     elseif (max(lmax) > 1)&(gc == maxshakes),
       'G1 Fails Repeatedly, Step Back'
        iter = iter-1;
        gc = 1;
     end
end

% sample from markov chain
SD = SA(:,:,D:D:NG);    % varname(1,:)     % draws of the state vector
QD = QA(:,:,D:D:NG);    % varname(2,:)     % draws of covariance matrix for state innovations
HD = H(:,:,D:D:NG);     % varname(3,:)     % stochastic volatilities
CD = CH(:,D:D:NG);      % varname(4,:)     % elements of the B matrix (cholesky factor)
VD = SV(:,D:D:NG);      % varname(5,:)     % standard error for volatility innovation
%RD = RP(:,:,D:D:NG);

save(DFILE(file,:),varname(1,:),varname(2,:),varname(3,:),varname(4,:),varname(5,:))
%save(DFILE(file,:),varname(1,:),varname(2,:),varname(3,:),varname(4,:),varname(5,:),varname(6,:))

% reinitialize gibbs arrays (buffer for back step)
SA(:,:,1:4) = SA(:,:,NG-3:NG);
QA(:,:,1:4) = QA(:,:,NG-3:NG);
H(:,:,1:4) = H(:,:,NG-3:NG);
CH(:,1:4) = CH(:,NG-3:NG);
SV(:,1:4) = SV(:,NG-3:NG);
%RP(:,:,1:4) = RP(:,:,NG-3:NG);

end
%exit

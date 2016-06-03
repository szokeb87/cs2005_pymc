  NG = 500; % number of draws in each file
  NF = 20; % number of files
  N = 3; % number of variables
  L = 2; % number of lags
  
% selection vectors 
%spi = [1 0 0 0 0 0]; % models 1-3, 6,7
%spi = [0 1 0 0 0 0]; % models 4,9
spi = [0 0 1 0 0 0]; % models 5,8
 
%su = [1 0 0 0 0 0]; % models 8,9
su = [0 1 0 0 0 0]; % models 1-3, 5,7
%su = [0 0 1 0 0 0]; % models 4,6

% catalog of data files
  DFILE(1,:) = ['\sims\mod5r\svol01'];
  DFILE(2,:) = ['\sims\mod5r\svol02'];
  DFILE(3,:) = ['\sims\mod5r\svol03'];
  DFILE(4,:) = ['\sims\mod5r\svol04'];
  DFILE(5,:) = ['\sims\mod5r\svol05'];
  DFILE(6,:) = ['\sims\mod5r\svol06'];
  DFILE(7,:) = ['\sims\mod5r\svol07'];
  DFILE(8,:) = ['\sims\mod5r\svol08'];
  DFILE(9,:) = ['\sims\mod5r\svol09'];
  DFILE(10,:) = ['\sims\mod5r\svol10'];
  DFILE(11,:) = ['\sims\mod5r\svol11'];
  DFILE(12,:) = ['\sims\mod5r\svol12'];
  DFILE(13,:) = ['\sims\mod5r\svol13'];
  DFILE(14,:) = ['\sims\mod5r\svol14'];
  DFILE(15,:) = ['\sims\mod5r\svol15'];
  DFILE(16,:) = ['\sims\mod5r\svol16'];
  DFILE(17,:) = ['\sims\mod5r\svol17'];
  DFILE(18,:) = ['\sims\mod5r\svol18'];
  DFILE(19,:) = ['\sims\mod5r\svol19'];
  DFILE(20,:) = ['\sims\mod5r\svol20'];

%  compute RA
  load(DFILE(11,:),'HD','CD')
  [T,N,P] = size(HD);
  HA = zeros(T,N,NF*P/2);
  CA = zeros(N,NF*P/2);
  HA(:,:,1:P) = HD;
  CA(:,1:P) = CD;
  clear HD CD
  for i = 12:NF,
    j = i - 10; 
    load(DFILE(i,:),'HD','CD')
    HA(:,:,(j-1)*P+1:j*P) = HD;
    CA(:,(j-1)*P+1:j*P) = CD;
    clear HD CD;
  end
  [T,N,NMC] = size(HA);
  RA = zeros(N,N,T-1,NMC);
  for i = 1:NMC,
    B = chofac(N,CA(:,i));
    IB = inv(B);
    for t = 2:T,
       RA(:,:,t-1,i) = IB*diag(HA(t,:,i))*IB';
    end
  end
    date = 1959 + [-1:1:T-2]'/4;
  ELH = mean(log(HA),3);
  EH = mean(HA,3);
  ER = mean(RA,4);
  clear CA HA RA B IB

% variance of orthogonalized innovations
  [T,N] = size(ELH);
  date = 1959 + [-1:1:T-2]'/4;
  figure;
  plot(date(6:T,1),ELH(6:T,:))
  ylabel('Elog(h)')
  axis([1960 2000 -inf inf])

% variance of VAR innovations
  [N,N,T] = size(ER);
  date = 1959 + [0:1:T-1]'/4;
  R11 = squeeze(ER(1,1,:));
  R21 = squeeze(ER(2,1,:));
  R22 = squeeze(ER(2,2,:));
  R31 = squeeze(ER(3,1,:));
  R32 = squeeze(ER(3,2,:));
  R33 = squeeze(ER(3,3,:));
  C21 = R21./((R11.*R22).^.5); % correlations
  C31 = R31./((R11.*R33).^.5);
  C32 = R32./((R22.*R33).^.5);

  figure;
  plot(date(8:4:T,1),log(R11(8:4:T,1)),'o-'); hold on;
  plot(date(8:4:T,1),log(R22(8:4:T,1)),'+-'); hold on;
  plot(date(8:4:T,1),log(R33(8:4:T,1)),'x-'); hold off;
  ylabel('Log of Variance')
  axis([1960 2000 -inf inf])

  figure;
  plot(date(8:4:T,1),C21(8:4:T,1),'o-'); hold on;
  plot(date(8:4:T,1),C31(8:4:T,1),'+-'); hold on;
  plot(date(8:4:T,1),C32(8:4:T,1),'x-'); hold off;
  ylabel('Correlations')
  axis([1960 2000 -inf inf])

% total prediction variance
  ldR = zeros(T,1);
  for t = 1:T,
    ldR(t,1) = log(det(ER(:,:,t)));
  end
  figure;
  plot(date(5:T,1),ldR(5:T,1)); 
  ylabel('log det R(t)')
  
  
% load SA
  load(DFILE(11,:),'SD')
  [NK,T,P] = size(SD);
  SA = zeros(NK,T,NF*P/2);
  SA(:,:,1:P) = SD;
  clear SD
  for i = 12:NF,
      j = i - 10; 
      load(DFILE(i,:),'SD')
      SA(:,:,(j-1)*P+1:j*P) = SD;
      clear SD
  end
  ES = mean(SA,3);
  clear SA
  [NK,T] = size(ES);
  
% core inflation and natural rate  
  S = [eye(N*(L-1)),zeros(N*(L-1),N)]; 
  mm  = zeros(N*L,T);
  for t = 1:T,
     A = [ES(2:1+N*L,t)'; ES(2+(1+N*L):2*(1+N*L),t)'; ES(2+2*(1+N*L):3*(1+N*L),t)';S];
     M = inv(eye(N*L) - A)*[ES(1,t); ES(2+N*L,t); ES(3+2*N*L,t); zeros(N*(L-1),1)];
     mm(:,t) = M;
  end
  core = (spi*mm)';
  lu = (su*mm)';
  nr = exp(100*lu)./(1+exp(100*lu)); % inverse logit for unemployment

  date = 1959 + [0:1:T-1]'/4;
  figure;
  plot(date(8:4:T,1),4*core(8:4:T,1),'o-'); hold on;
  plot(date(8:4:T,1),nr(8:4:T,1),'+-'); hold off;
  ylabel('Percent')
  axis([1960 2000 0 .1])
  
% frequency labels
  TT = T;
  TT2 = ceil(TT/2);
  jj = [0:1:TT2];
  ww = 2*pi*jj/TT; % radians per quarter
  cc = jj/TT; % cycles per quarter
  ccy = 4*jj/TT; % cycles per year

% initializing the fourier frequencies
  w = zeros(TT,1);
  im = (-1)^.5;
  for j = 1:TT,
     w(j,1) = exp(-2.0*im*pi*(j-1)/TT);  
  end 

% local linear spectral density
  S = [eye(N*(L-1)),zeros(N*(L-1),N)];
  IA = eye(N*L); % IA is an identity matrix conformable with A
  A = zeros(N*L,N*L); % VAR companion matrix
  B = zeros(N*L,N*L); % innovation variance
  %C = [eye(N),zeros(N,N*L-N)]; % a selection matrix
  %D = zeros(N,N*L);

  F11 = zeros(TT2,T); % spectrum for inflation
  G11 = zeros(TT2,T); % normalized spectrum for inflation
  v11 = zeros(T,1); % variance of inflation
  for t = 1:T
     A = [ES(2:1+N*L,t)'; ES(2+(1+N*L):2*(1+N*L),t)'; ES(2+2*(1+N*L):3*(1+N*L),t)';S];
     B(1:N,1:N) = ER(:,:,t);	
     Vyy = doublej(A,B);
     v11(t,1) = spi*Vyy*spi'; 
     for j = 1:TT2+1,
       TR = inv(IA - A*w(j,1)); 
       G = TR*B*TR'/(2.0*pi);
       F11(j,t) = real(spi*G*spi');
       G11(j,t) = F11(j,t)/v11(t,1);
     end 
  end
  
% spectrum for inflation
  figure;
  colormap(white);
  surf(F11(:,12:4:T)');
  axis([0 inf 0 inf 0 inf]);
  xlabel('Cycles per Quarter')
  ylabel('Year')
  zlabel('Power')
  %title('Spectrum for Inflation')
  set(gca,'xticklabel',[0 .116 .233 .349 .465])
  set(gca,'yticklabel',[1960 1970 1980 1990 2000])
  view([50 30])

% normalized spectrum for inflation
  figure;
  colormap(white);
  surf(G11(:,12:4:T)');
  axis([0 inf 0 inf 0 inf]);
  xlabel('Cycles per Quarter')
  ylabel('Year')
  zlabel('Power')
  %title('Normalized Spectrum for Inflation')
  set(gca,'xticklabel',[0 .116 .233 .349 .465])
  set(gca,'yticklabel',[1960 1970 1980 1990 2000])
  view([50 30])
 
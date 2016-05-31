% xmat.m

% construct the (K1 +k2 +...kn) by N by T matrix X called by 
% function file recadp.m

function [X] = xmat(X1,X2,X3)

% Xi is the ki by T matrix of the RHS used in equation i. K1 is the 
% number of RHS varaibles, T denotes time periods

% We have to adjust X!, X2, X3 manually for different cases.

N = 3; % N is the number of equations, currently 3: X1, X2 and X3.

[rx1,cx1] = size(X1);
[rx2,cx2] = size(X2);
[rx3,cx3] = size(X3);

if cx1 ~= cx2 | cx2 ~= cx3
   disp('The time periods of the series not the same, check!')
end

T = cx1;

data = [X1;X2;X3];
rd = size(data,1);

onemat = zeros(rd,3);

onemat(1:rx1,1) = ones(rx1,1);
onemat(rx1+1:rx1+rx2,2) = ones(rx2,1);
onemat(rx1+rx2+1:rd,3) = ones(rx3,1);

for i = 1:T
   
   temp(:,:,i) = ndgrid(data(:,i),ones(1,N));
   X(:,:,i) = temp(:,:,i).*onemat;
   
end

   

cont = zeros(50000,1);
cont2 = zeros(5000,1);

for i = 1:50000
    cont(i,:)=trace(GIBBS2Q(Q0,df,3,2));
end
%M=squeeze(mean(cont,1))
mean(cont)

function [Phi,u]=create_dictionary(m,n,type)
if strcmp(type,'random')
    Phi=randn(m,n);
    u=zeros(1,n);
elseif strcmp(type,'ULA')
    u=(-1:(2/n):(1-2/n));
    Phi=exp(-1j*(0:m-1)'*pi*u);
end
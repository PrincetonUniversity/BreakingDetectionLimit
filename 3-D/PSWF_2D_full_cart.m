function [psi_Nn, n_list] = PSWF_2D_full_cart(maxN, L, beta, T)
% Function to generate 2D PSWFs, evaluated on a Cartesian grid. We include
% negative angular frequency PSWFs as well.
% 
% Inputs:
%   * maxN: maximum angular frequency for PSWFs. If left empty, maxN is
%   chosen by the truncation parameter T (see below)
%   * L: length of volume or projection
%   * beta: fraction of Nyquist assumed to be the bandlimit
%   * T: truncation parameter determining length of PSWF expansion
% 
% Outputs:
%   * psi_Nn: cell array sorted by angular frequency (from negative to
%   positive) of PSWFs evaluated on a cartesian grid of size L
%   * n_list: list of number of radial frequencies for each angular
%   frequency
% 
% Eitan Levin, July 2018

Lr = floor(L/2);
n_list = precomp_Nn_list(Lr, beta, T);
if isempty(maxN)
    maxN = length(n_list)-1;
end
c = beta*pi*Lr;

if mod(L,2) > 0
    [x, y] = meshgrid(-Lr:Lr, -Lr:Lr);
else
    [x, y] = meshgrid(-Lr:Lr-1, -Lr:Lr-1);
end

[phi, r] = cart2pol(x./Lr,y./Lr);

psi_Nn = cell(2*maxN+1, 1);
for N = -maxN:maxN
    if n_list(abs(N)+1) > 1
        R_Nn = PSWF_radial_2D(abs(N), n_list(abs(N)+1)-1, c, r(r<=1));
    else
        R_Nn = PSWF_radial_2D(abs(N), n_list(abs(N)+1), c, r(r<=1));
        R_Nn = R_Nn(:,1);
    end
    psi_Nn{N+maxN+1} = zeros(L^2, n_list(abs(N)+1));
    psi_Nn{N+maxN+1}(r<=1, :) = bsxfun(@times, R_Nn, exp(1i*N*phi(r<=1))./sqrt(2*pi));
end
    
    

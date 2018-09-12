function [T, ang_freq] = precomp_shifted_PSWF_coeffs_incl_neg(psi_Nn, n_list, L, beta_PSWF, Trunc)
% Compute inner products between shifted and centered PSWFs (of bandlimit c 
% and 2c, respectively, where c = beta_PSWF*pi*L, see below) including 
% negative angular frequencies needed for the trispectrum.
% 
% Inputs:
%   * psi_Nn: cell array of PSWFs (see PSWF_2D_full_cart.m)
%   * n_list: list of number of radial frequencies for each angular
%   frequency of shifted PSWFs(see precomp_Nn_list.m)
%   * L: length of volume or projection
%   * beta_PSWF: fraction of Nyquist to assume as the bandlimit
%   * Trunc: truncation parameter controlling length of PSWF expansion
% 
% Outputs:
%   * T: cell array of inner products
%   * ang_freq: list of number of radial frequencies per angular frequency
%   of centered PSWFs.
% 
% Eitan Levin, August 2018

[Mt, ang_freq] = precomp_pswf_t_mat_direct(2*L-1, beta_PSWF, Trunc);
num_freqs = length(ang_freq);
[x,y] = meshgrid(-L+1:L-1, -L+1:L-1); pts_notin_disc = sqrt(x.^2 + y.^2) > L-1;
Mt(:, pts_notin_disc) = 0;

maxN = floor(length(psi_Nn)/2);
T = cell(2*maxN+1, 1);
for N = -maxN:maxN
    n_curr = n_list(abs(N)+1);
    psi_Nn{N+maxN+1} = reshape(psi_Nn{N+maxN+1}, L, L, n_curr);
    psi_Nn{N+maxN+1} = padarray(psi_Nn{N+maxN+1}, [L-1, L-1]);
    
    patch = zeros(2*L-1, 2*L-1, n_curr, L^2);
    for col = L:2*L-1
        for row = L:2*L-1
            idx = (row-L+1) + (col-L)*L;
            patch(:,:,:,idx) = psi_Nn{N+maxN+1}(row-L+1:row+L-1, col-L+1:col+L-1, :);
        end
    end
    patch = reshape(patch, (2*L-1)^2, n_curr*L^2);
            
    T{N+maxN+1} = Mt*patch; % compute PSWF expansion coefficients
    T{N+maxN+1} = reshape(T{N+maxN+1}, num_freqs, n_curr, L^2);
    T{N+maxN+1} = reshape(permute(T{N+maxN+1}, [3,1,2]), L^2*num_freqs, n_curr);
end
           

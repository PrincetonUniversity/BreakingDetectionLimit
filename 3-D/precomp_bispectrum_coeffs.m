function [B, L1_list, L2_list, L3_list, blk_id] = precomp_bispectrum_coeffs(maxL, s_list, L, W)
% Function to precompute the coefficients for evaluation of the bispectrum
% given volume expansion coefficients.
% 
% Inputs:
%   * maxL: spherical harmonics cutoff
%   * s_list: list of radial frequencies for each order of spherical
%   harmonics
%   * L: length of volume (or projection)
%   * W: precomputed weights proportional to Wigner 3j symbol (see
%   precomp_wigner_weights.m)
% 
% Outputs:
%   * B: the precomputed coefficients
%   * L1_list, L2_list, L3_list: lists of spherical harmonics orders for
%   vectorization of B
%   * blk_id: angular frequency list for the vectorized bispectrum
% 
% Eitan Levin, July 2018

beta_PSWF = 1; Trunc = 1e-1;
[Mt, ang_freq] = precomp_pswf_t_mat(2*L-1, beta_PSWF, Trunc);
[x,y] = meshgrid(-L+1:L-1, -L+1:L-1); pts_notin_disc = sqrt(x.^2 + y.^2) > L-1;
Mt(:,pts_notin_disc(:)) = 0;

[psi_Nn, n_list] = PSWF_2D_full_cart(maxL, L, beta_PSWF, 1e-5);
beta = sph_Bessel_to_2D_PSWF_factors(maxL, n_list, s_list(1), floor(L/2)); % l x N x s x n

psi_lNs = cell(maxL+1,1);
for l = 0:maxL
    psi_lNs{l+1} = cell(2*l+1, 1); % N
    for N = -l:l
        psi_lNs{l+1}{N+l+1} = psi_Nn{N+maxL+1}*beta{l+1}{abs(N)+1}(1:s_list(l+1),:).';
        if N < 0
            psi_lNs{l+1}{N+l+1} = (-1)^(N)*psi_lNs{l+1}{N+l+1};
        end
        psi_lNs{l+1}{N+l+1} = reshape(psi_lNs{l+1}{N+l+1}, L, L, []);
        psi_lNs{l+1}{N+l+1} = icfft2(psi_lNs{l+1}{N+l+1});
        psi_lNs{l+1}{N+l+1} = padarray(psi_lNs{l+1}{N+l+1}, [L-1, L-1]);
    end
end

% clear psi_Nn beta n_list
maxN = max(ang_freq);
q_list = zeros(maxN+1, 1);
blk_id = [];
for ii = 1:length(q_list)
    q_list(ii) = sum(ang_freq == ii-1);
    blk_id(end+1:end+q_list(ii)^2, 1) = ii;
end
q_cumsum = cumsum([0; q_list(:)]);
q_sq_cumsum = cumsum([0; q_list(:).^2]);
kq_len = length(blk_id);

L1_list = []; L2_list = []; L3_list = [];
for ii = 1:(maxL+1)^2
    [L1, L2] = ind2sub([maxL+1, maxL+1], ii);
    L1 = L1-1; L2 = L2-1;
    
    L3_vals = abs(L1-L2):min(L1+L2, maxL);
    for jj = 1:length(L3_vals)
        L1_list(end+1) = L1;
        L2_list(end+1) = L2;
        L3_list(end+1) = jj;
    end
end

vec = @(x) x(:);
B = cell(length(L1_list), 1);
num_freqs = length(ang_freq);

parfor ll = 1:length(L1_list)
    L1 = L1_list(ll);
    L2 = L2_list(ll);
    ii_3 = L3_list(ll); L3 = abs(L1-L2) + ii_3 - 1;
    
    s1_len = s_list(L1+1);
    s2_len = s_list(L2+1);
    s3_len = s_list(L3+1);
    
    acc_vec = zeros(kq_len*s3_len*s2_len,s1_len, 'double');
    
    for N1 = -L1:L1
        for N2 = max(-L2, -L3-N1):min(L2, L3-N1)
            N3 = N1+N2;
            norm_fact = W{L1+1, L2+1}(N2+L2+1, N1+L1+1, ii_3)/L^2;
            patch = zeros(2*L-1, 2*L-1, s2_len + s3_len, L^2);
            for col = L:2*L-1
                for row = L:2*L-1
                    idx = (row-L+1) + (col-L)*L;
                    patch(:, :, 1:s2_len, idx) = psi_lNs{L2+1}{N2+L2+1}(row-L+1:row+L-1, col-L+1:col+L-1, :);
                    patch(:, :, s2_len+1:end, idx) = psi_lNs{L3+1}{N3+L3+1}(row-L+1:row+L-1, col-L+1:col+L-1, :);
                end
            end
            patch = reshape(patch, (2*L-1)^2, (s2_len + s3_len)*L^2);
            
            T = Mt*patch; % compute PSWF expansion coefficients
            T = reshape(T, num_freqs, s2_len+s3_len, L^2);
           
            for col = L:2*L-1
                T_acc = zeros(kq_len*s3_len*s2_len, L, 'double');
                P1 = reshape(psi_lNs{L1+1}{N1+L1+1}(L:2*L-1, col, :), L, s1_len);
                for row = L:2*L-1
                    T_add = zeros(kq_len,s3_len*s2_len,'double');
                    idx = (row-L+1) + (col-L)*L;
                    for N = 0:maxN
                        T_N = T(q_cumsum(N+1)+1: q_cumsum(N+2), :, idx);
                        T_N = vec(T_N(:,1:s2_len))*vec(T_N(:,s2_len+1:end))'; % (q1, s2) x (q2, s3)
                        T_N = permute(reshape(T_N, q_list(N+1), s2_len, q_list(N+1), s3_len), [1,3,4,2]); % (q1, q2, s3, s2)
                        T_add(q_sq_cumsum(N+1)+1: q_sq_cumsum(N+2), :) = T_add(q_sq_cumsum(N+1)+1: q_sq_cumsum(N+2), :) + ...
                            reshape(T_N, q_list(N+1)^2, s3_len*s2_len);
                    end
                    T_acc(:, row-L+1) = T_add(:);
                end
                acc_vec = acc_vec + norm_fact*T_acc*P1; % (q1, q2, s3, s2) x s1
            end
            
        end
    end
    B{ll} = reshape(acc_vec, kq_len, s3_len, s2_len, s1_len); % (q1, q2) x s3 x s2 x s1
end


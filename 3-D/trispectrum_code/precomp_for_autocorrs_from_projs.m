function [psi_coeffs_lNs, psi_lNs, q_list, D_mats, a_sizes, ang_freq] = ...
                                         precomp_for_autocorrs_from_projs(maxL, s_list, L, Rots)
% Precomputations for the computation of autocorrelations from projections.
% 
% Inputs:
%   * maxL: cutoff for spherical harmonics expansion
%   * s_list: list of number of radial frequencies for each order of
%   spherical harmonics
%   * L: length of volume or projection
%   * Rots: stack of 3x3 rotation matrices representing the viewing
%   directions for the projections
% 
% Outputs:
%   * psi_coeffs_lNs: inner products between (linear combinations of) 
%   shifted PSWFs of bandlimit c and centered PSWFs of bandlimit 2c.
%   * psi_lNs: linear combination of centered PSWFs
%   * q_list: list of number of radial frequencies per angular frequency of
%   PSWFs
%   * D_mats: Wigner-D matrices corresponding to Rots
%   * a_sizes: number of volume expansion coefficients for each order of
%   spherical harmonics
%   * ang_freq: list of the angular frequencies corresponding to each entry
%   of psi_coeffs_lNs
% 
% Eitan Levin, August 2018

% Generate PSWFs:
beta_PSWF = 1; Trunc_img = 1e-5; Trunc_bispect = 1e-1;
[psi_Nn, n_list] = PSWF_2D_full_cart(maxL, L, beta_PSWF, Trunc_img);
psi_Nn = cellfun(@(x) reshape(icfft2(reshape(x, L, L, [])), L^2, []), psi_Nn, 'UniformOutput', 0);

% Compute inner products between shifted and centered PSWFs:
[T, ang_freq] = precomp_shifted_PSWF_coeffs_incl_neg(psi_Nn, n_list, L, beta_PSWF, Trunc_bispect);
num_freqs = length(ang_freq);
maxN = max(ang_freq);
q_list = zeros(2*maxN + 1, 1);
for N = -maxN:maxN
    q_list(N+maxN+1) = sum(ang_freq == N);
end

a_sizes = s_list; % numel of a_lms
for ii = 0:maxL 
    a_sizes(ii+1) = (2*ii+1)*a_sizes(ii+1); 
end

beta = sph_Bessel_to_2D_PSWF_factors(maxL, n_list, s_list(1), floor(L/2)); % l x N x s x n

% Compute linear combinations of PSWFs and their inner products for
% autocorrelation computations:
psi_lNs = cell(maxL+1, 1);
psi_coeffs_lNs = cell(maxL+1, 1);
for l = 0:maxL
    for N = -l:l
        psi_lNs{l+1}{N+l+1} = (-1)^(N*(N<0))*psi_Nn{N+maxL+1}...
                                                    *beta{l+1}{abs(N)+1}(1:s_list(l+1), :).';
        psi_coeffs_lNs{l+1}{N+l+1} = (-1)^(N*(N<0))*T{N+maxL+1}...
                                                    *beta{l+1}{abs(N)+1}(1:s_list(l+1), :).';
    end
    psi_lNs{l+1} = cat(2, psi_lNs{l+1}{:});
    psi_coeffs_lNs{l+1} = cat(2, psi_coeffs_lNs{l+1}{:});
end

psi_lNs = cat(2, psi_lNs{:});
psi_lNs = mat2cell(psi_lNs, L^2, a_sizes);
psi_lNs = psi_lNs(:);
psi_lNs = cellfun(@(x,y) reshape(x, [], y), psi_lNs, num2cell(a_sizes./s_list), 'UniformOutput', 0);

psi_coeffs_lNs = cat(2, psi_coeffs_lNs{:});
psi_coeffs_lNs = reshape(psi_coeffs_lNs, L^2, num_freqs, []);
psi_coeffs_lNs = mat2cell(psi_coeffs_lNs, L^2, q_list, a_sizes);
psi_coeffs_lNs = permute(psi_coeffs_lNs, [3,2,1]);
psi_coeffs_lNs = cellfun(@(x,y) reshape(x, [], y), psi_coeffs_lNs, repmat(num2cell(a_sizes./s_list), 1, length(q_list)), 'UniformOutput', 0);

a_sizes = num2cell(a_sizes);

% Precompute Wigner-D matrices:
D_mats = cell(maxL + 1, size(Rots, 3));
for ii = 1:size(Rots, 3)
    D_mats(:, ii) = RN2RL(getSHrotMtx(Rots(:,:,ii), maxL, 'complex'));
end
D_mats = cellfun(@transpose, D_mats, 'UniformOutput', 0);

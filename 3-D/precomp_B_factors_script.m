% Script to precompute the coefficients for bispectrum evaluation

%% Paths:
addpath('PATH TO kam_Cryo')
addpath('PATH TO SPHERICAL HARMONICS TOOLBOX')
addpath('PATH TO EASYSPIN')
addpath('PATH TO ASPIRE')
initpath

%% Parameters
maxL = 2; % cutoff for spherical harmonics
L = 31; % length of volume (or projection)
r_cut = 1/2; % assumed cutoff (Nyquist is 1/2)

%% Actual computation:
s_list = gen_s_list(maxL, r_cut, floor(L/2)); % list of radial frequencies
W = precomp_wigner_weights(maxL); % weights proportional to Wigner 3j symbols

if isempty(gcp('nocreate')), parpool('local', maxNumCompThreads); end
[B, L1_list, L2_list, L3_list, blk_id] = precomp_bispectrum_coeffs(maxL, s_list, L, W);

B_lists.L1 = L1_list;
B_lists.L2 = L2_list;
B_lists.L3 = L3_list;
B_lists.blk_id = blk_id;

save(['B_factors_bispect_maxL' num2str(maxL) '_L' num2str(L) '.mat'], '-v7.3')

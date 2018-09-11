% Script to load a volume, generate its first four autocorrelations, and
% inverts them to reconstruct the volume.
%% Paths
clear all; close all;
addpath('PATH TO kam_cryo')
addpath('PATH TO SPHERICAL HARMONICS TOOLBOX')
addpath('PATH TO ASPIRE')
initpath
addpath('PATH TO MANOPT')

%% Basic parameters and volume

info.maxL = 5; % truncation for spherical harmonics expansion
info.r_cut = 1/2; % assumed bandlimit (Nyquist is 1/2)
info.L0 = 21; % length of volume
info.N=floor(info.L0/2);

vol = double(ReadMRC('1qlq_crop.mrc')); % load volume
vol = cryo_downsample(vol, info.L0); % downsample if necessary

% Expand volume:
[Psilms, Psilms_2D, jball, jball_2D] = precompute_spherical_basis(info); % compute volume basis
[a_lms, vol_true_trunc] = expand_vol_spherical_basis(vol, info, Psilms, jball); % expand volume
vol_true_trunc = real(icfftn(vol_true_trunc)); % ground truth volume

% Vectorize coefficients and generate lists of indices for the vector:
x_lists.s = gen_s_list(info.maxL, info.r_cut, 1, floor(info.L0/2));
[x_lists.L, x_lists.m, ~, x_lists.p, x_lists.n] = gen_vec_coeff_lists(info.maxL, x_lists.s);
a_vec = vec_cell(a_lms);
%% Generate rotation matrices for autocorrelation estimation from projections
rng(10) % set seed
num_pts = 120 % number of projections to use for autocorrelation estimation
% generate rotations corresponding to an approximately uniform mesh on the 
% sphere and a random in-plane rotation:
Rots = sample_S2(num_pts); 

%% Compute true autocorrelations:

% Precompute quantities for trispectrum computation from projections:
[M4_quants.psi_coeffs_lNs, M4_quants.psi_lNs, M4_quants.q_list, ...
    M4_quants.D_mats, M4_quants.a_sizes, M4_quants.ang_freq] = ...
    precomp_for_autocorrs_from_projs(info.maxL, x_lists.s, info.L0, Rots);

k_max = 5; % maximum angular frequency for the trispectrum
maxK = (length(M4_quants.q_list) - 1)/2; % maximum possible angular frequency given our cutoffs
M4_quants.psi_coeffs_lNs = M4_quants.psi_coeffs_lNs(:,-2*k_max+maxK+1:2*k_max+maxK+1); % keep only those quantities that appear in the computation

if isempty(gcp('nocreate')), parpool(maxNumCompThreads); end

% Compute actual trispectrum:
[~, m4_micro] = trispectrum_grad_from_harmonics_projs_projPar(a_lms,...
    M4_quants.psi_coeffs_lNs, M4_quants.psi_lNs, M4_quants.q_list,...
    M4_quants.D_mats, info.L0, k_max);

% Compute bispectrum:
load('/scratch/network/eitanl/B_factors_bispect_maxL5_L21.mat', 'B', 'B_lists', 'W')

[G1, G2, G3] = bispectrum_grad_from_harmonics...
        (a_lms, B, W, B_lists.L1, B_lists.L2, B_lists.L3, x_lists.L, x_lists.m, x_lists.s, B_lists.blk_id);
    
m3_micro = real(G1.'*a_vec);

% Compute mean and power spectrum:
q0 = sqrt(sum(blk_id == 1));
[r,w] = lgwt(40*q0, 0, 1);
M12_quants.w = w.*r;

M12_quants.j_l = generate_spherical_bessel_basis(info.maxL, x_lists.s, info.r_cut, info.r_cut*r);
M12_quants.j_0 = cell2mat(generate_spherical_bessel_basis(0, x_lists.s, info.r_cut, 0));

[R_0n, alpha_Nn_2D] = PSWF_radial_2D(0, q0-1, pi*(info.L0-1), r); % generate 2D radial prolates
M12_quants.R_0n = bsxfun(@times, R_0n, 2./alpha_Nn_2D(:).');

m2_micro = power_spectrum_from_harmonics(a_vec, M12_quants.j_l, M12_quants.R_0n, M12_quants.w, info.L0, info.maxL);

m1_micro = mean_from_harmonics(a_vec, M12_quants.j_0, x_lists.s(1), info.L0);

%% Invert autocorrelations to recover volume
a_init = []; % initial guess for coefficients, leave empty to use random initialization
[a_lms_rec, gamma_rec] = recover_vol_coeffs_from_moments_projs_trispect...
    (m4_micro, m3_micro, m2_micro, m1_micro, M4_quants, B, W, B_lists,...
    M12_quants, x_lists, info.maxL, info.L0, Rots, k_max, a_init);

a_aligned = align_vol_coeffs(a_lms_rec, a_lms, 1e5, 1); % rotationally align recovered coefficients to true ones

norm(vec_cell(a_aligned) - a_vec)/norm(a_vec)

abs(gamma_rec - 1)

% Generate recovered and aligned volume:
vol_rec_aligned = recover_from_ALM_v4_given_Psilms(a_aligned, info.N, jball, Psilms, info.maxL, info.L0);

vol_rec_aligned = real(icfftn(vol_rec_aligned));

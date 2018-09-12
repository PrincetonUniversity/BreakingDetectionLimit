% Script to load volume, compute the first three full autocorrelations and
% a 4-tensor slice through the 5-tensor trispectrum. Here we use
% projections to compute both bispectrum and trispectrum slice.
%% Paths
clear all; close all;
addpath('PATH TO kam_cryo')
addpath('PATH TO SPHERICAL HARMONICS TOOLBOX')
addpath('PATH TO ASPIRE')
initpath
addpath('PATH TO MANOPT TOOLBOX')

%% Basic parameters and volume

% Basic parameters and volume
info.maxL = 7; % truncation for spherical harmonics expansion
info.r_cut = 1/2; % assumed bandlimit (Nyquist is 1/2)
info.L0 = 21; % size of volume
info.N=floor(info.L0/2);

vol = double(ReadMRC('1qlq_crop.mrc')); % load volume
vol = cryo_downsample(vol, info.L0); % downsample if necessary

% Expand volume:
[Psilms, Psilms_2D, jball, jball_2D] = precompute_spherical_basis(info);
[a_lms, vol_true_trunc] = expand_vol_spherical_basis(vol, info, Psilms, jball); % expand volume
vol_true_trunc = real(icfftn(vol_true_trunc)); % ground truth volume

% Vectorize coefficients and generate lists of indices for the vector:
x_lists.s = gen_s_list(info.maxL, info.r_cut, 1, floor(info.L0/2));
[x_lists.L, x_lists.m, ~, x_lists.p, x_lists.n] = gen_vec_coeff_lists(info.maxL, x_lists.s);
a_vec = vec_cell(a_lms);

%% Generate rotation matrices for autocorrelation estimation from projections
rng(10)
num_pts = 100;
Rots = sample_S2(num_pts);

%% Compute true autocorrelations:

% Precompute quantities for bispectrum and trispectrum slice (mainly inner
% products of shifted PSWFs with centered ones):
[psi_curr, curr_freqs, psi_lNs, q_list, D_mats, psi_freqs, a_sizes, psi_curr_k0] = ...
    precomp_for_autocorrs_from_projs_GPU(info.maxL, x_lists.s, info.L0, Rots);

[G4, M4] = trispectrum_4tensor_grad_from_harmonics_projs(a_vec, psi_curr, psi_curr_k0, curr_freqs, psi_lNs, q_list, D_mats, L, psi_freqs, a_sizes, maxL);

m4_micro = vec_cell(M4);

[~, M3] = bispectrum_grad_from_harmonics_projs(a_vec, psi_curr, curr_freqs, psi_lNs, q_list, D_mats, L, psi_freqs, a_sizes, maxL);

m3_micro = vec_cell(M3);

q0 = q_list(1);
[r,w] = lgwt(40*q0, 0, 1);
M12_quants.w = w.*r;

M12_quants.j_l = generate_spherical_bessel_basis(maxL, s_list, 1/2, (1/2)*r);
M12_quants.j_0 = cell2mat(generate_spherical_bessel_basis(0, s_list, 1/2, 0));

[R_0n, alpha_Nn_2D] = PSWF_radial_2D(0, q0-1, pi*(L-1), r); % generate 2D radial prolates
M12_quants.R_0n = bsxfun(@times, R_0n, 2./alpha_Nn_2D(:).');

m2_micro = power_spectrum_from_harmonics(a_vec, M12_quants.j_l, M12_quants.R_0n, M12_quants.w, L, maxL);

m1_micro = mean_from_harmonics(a_vec, M12_quants.j_0, s_list(1), L);

[a_lms_rec, gamma_rec] = recover_vol_coeffs_from_moments_projs_trispect_4tensor...
                    (m4_micro, m3_micro, m2_micro, m1_micro, maxL, L, r_cut, [], Rots);

a_aligned = align_vol_coeffs(a_lms_rec, a_lms, 1e5, 1);

norm(vec_cell(a_aligned) - a_vec)/norm(a_vec)

abs(gamma_rec - 1)

vol_rec_aligned = recover_from_ALM_v4_given_Psilms(a_aligned, info.N, jball, Psilms, info.maxL, info.L0);

vol_rec_aligned = real(icfftn(vol_rec_aligned));

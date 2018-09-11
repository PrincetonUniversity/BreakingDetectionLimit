% Script to generate the first three autocorrelations of a given volume and
% invert them to recover the volume back.

%% Paths (MODIFY AS NEEDED!)
clear all; close all;
addpath('PATH TO kam_cryo REPO')
addpath('PATH TO SPHERICAL HARMONIC TOOLBOX')
addpath('PATH TO PSWF TOOLBOX')
addpath('PATH TO ASPIRE TOOLBOX')
initpath

%% Basic parameters and volume
info.maxL = 5; % truncation for spherical harmonics expansion
info.r_cut = 1/2; % assumed bandlimit (1/2 is Nyquist)
info.L0 = 20; % size of volume
info.N=floor(info.L0/2);

vol = double(ReadMRC('1qlq_crop.mrc')); % load volume
vol = cryo_downsample(vol, L); % downsample if needed

% Expand volume:
[Psilms, Psilms_2D, jball, jball_2D] = precompute_spherical_basis(info); % generate basis
[a_lms, vol_true_trunc] = expand_vol_spherical_basis(vol, info, Psilms, jball); % expand
vol_true_trunc = real(icfftn(vol_true_trunc)); % ground truth volume

%% Generate clean autocorrelations from volume

% Vectorize volume expansion coefficients and generate lists giving various
% indices for the resulting vector
a_vec = vec_cell(a_lms);
x_lists.s = gen_s_list(info.maxL, info.r_cut, floor(info.L0/2));
[x_lists.L, x_lists.m, ~, x_lists.p, x_lists.n] = gen_vec_coeff_lists(info.maxL, x_lists.s);

% Load precomputed quantities for bispectrum evaluation:
load(['B_factors_bispect_maxL' num2str(info.maxL) '_L' num2str(info.L0) '.mat'], 'B', 'B_lists', 'W')

[G1, G2, G3] = bispectrum_grad_from_harmonics...
        (a_lms, B, W, B_lists.L1, B_lists.L2, B_lists.L3, x_lists.L, x_lists.m, x_lists.s, B_lists.blk_id);
    
% Compute quantities for mean and power spectrum evaluations:
q0 = sqrt(sum(blk_id == 1));
[r,w] = lgwt(40*q0, 0, 1);
M12_quants.w = w.*r;

M12_quants.j_l = generate_spherical_bessel_basis(info.maxL, x_lists.s, info.r_cut, info.r_cut*r);
M12_quants.j_0 = cell2mat(generate_spherical_bessel_basis(0, x_lists.s, info.r_cut, 0));

[R_0n, alpha_Nn_2D] = PSWF_radial_2D(0, q0-1, pi*(info.L0-1), r); % generate 2D radial prolates
M12_quants.R_0n = bsxfun(@times, R_0n, 2./alpha_Nn_2D(:).');
    
% Evaluate first three autocorrelations
a3_micro = real(G1.'*a_vec);

a2_micro = power_spectrum_from_harmonics(a_vec, M12_quants.j_l, M12_quants.R_0n, M12_quants.w, info.L0, info.maxL);

a1_micro = mean_from_harmonics(a_vec, M12_quants.j_0, x_lists.s(1), info.L0);

%% Invert autocorrelations to reconstruct volume
a_init = []; % initial guess for coefficients - if empty, generates random initialization

[a_lms_rec, gamma_rec] = recover_vol_coeffs_from_moments(a3_micro, a2_micro, a1_micro, info.maxL, info.L0, B, W, B_lists, info.r_cut, a_init);

a_aligned = align_vol_coeffs(a_lms_rec, a_lms, 1e5, 1); % align reconstructed and true volumes

disp('Error in expansion coefficients:')
norm(vec_cell(a_aligned) - a_vec)/norm(a_vec)

disp('Error in gamma:')
abs(gamma_rec - 1)

% Form reconstructed and aligned volume:
vol_rec_aligned = recover_from_ALM_v4_given_Psilms(a_aligned, info.N, jball, Psilms, info.maxL, info.L0);

vol_rec_aligned = real(icfftn(vol_rec_aligned));

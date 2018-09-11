% Script to recover number of projections per micrograph from
% autocorrelations

%% Paths
clear all; close all;
addpath('PATH TO kam_cryo REPO')
addpath('PATH TO SPHERICAL HARMONICS TOOLBOX')
addpath('PATH TO ASPIRE TOOLBOX')
initpath
addpath('PATH TO MANOPT')

%% Parameters and precomputations:
% Basic parameters and volume
info.maxL = 0; % truncation for spherical harmonics expansion
info.r_cut = 1/2;
info.L0 = 31; % size of volume
info.N=floor(info.L0/2);

load(['B_factors_bispect_maxL' num2str(info.maxL) '_L' num2str(info.L0) '.mat'], 'B', 'B_lists', 'W')

load('MOMENTS FROM MICROGRAPH', 'm1', 'm2', 'm3', 'M')

if isempty(gcp('nocreate')), parpool(maxNumCompThreads); end

%% Recover number of projections:
a_init = [];
[a_lms_rec, gamma_rec] = recover_vol_coeffs_from_moments(m3, m2, m1, info.maxL,...
    info.L0, B, W, B_lists, info.r_cut, a_init);

num_projections = gamma_rec*M^2/info.L0^2;
% Script to compute moments of pure noise micrographs

%% Set paths:
clear all; close all;
addpath('PATH TO kam_cryo toolbox')
addpath('PATH TO SPHERICAL HARMONICS TOOLBOX')
addpath('PATH TO ASPIRE')
initpath

%% Set parameters, load micrograph
sigma = 25; % variance of the noise
seed = 19; % seed for noise generation
rng(seed)
maxNumCompThreads(4); % set number of workers

M = 7420; % length of micrograph
L = 31; % assumed length of volume (or projection)
num_micros = 25; % number of micrographs

%% Compute the moments:

I = sigma*randn(M, M, num_micros);

% Compute moments:
disp('Computing moments')
pad_len = 256 - 2*(L-1);
[m1, m2, m3] = moments_from_micrograph_steerable(I, L, pad_len);
    
% Debias moments:
disp('Debiasing')
[m2, m3] = debias_moments_steerable(m1, m2, m3, sigma, L);

disp('Saving:')

filename = ['pure_noist_test_microNum_' num2str(num_micros) '_M_' num2str(M) '_L_' num2str(L) '_sigma_' num2str(sigma) '.mat'];
disp(['Filename: ' filename])

save(filename, 'm1', 'm2', 'm3', 'seed', 'sigma', 'M')

% Script to load a stack of micrographs in groups and compute the
% autocorrelations of the dataset.
%% Set paths:
clear all; close all;
addpath('PATH TO kam_cryo TOOLBOX')
addpath('PATH TO SPHERICAL HARMONICS TOOLBOX')
addpath('PATH TO ASPIRE TOOLBOX')
initpath

%% Set parameters, load micrograph
% SNR = 1/1024;
% SNR = inf;
sigma = 25;
seed = 15; % seed for noise generation
rng(seed)
maxNumCompThreads(4); % set number of workers

disp('Preparing .mat file')
img_file = matfile('MICROGRAPH FILES');

L = img_file.L;
gamma_true = img_file.gamma;
M = img_file.M;
num_micros = img_file.num_micros;

%% Process micrographs in batches:

micros_per_batch = 100 % micrographs per read
batch_num = ceil(num_micros/micros_per_batch)
m1 = 0; m2 = 0; m3 = 0;
for ii = 1:batch_num
    start_idx = (ii-1)*micros_per_batch + 1;
    end_idx = min(ii*micros_per_batch, num_micros);

    disp(['Reading micrographs ' num2str(start_idx) ' to ' num2str(end_idx)])
    I = img_file.I(start_idx:end_idx, 1);
    I = cat(3, I{:});
    I = padarray(I, [L-1, L-1]);

    % Add noise:
    disp(['Adding noise with sigma = ' num2str(sigma)])
%    [I, ~, ~, sigma_true] = cryo_addnoise(I, SNR, 'gaussian');
    SNR = var(I(:))/sigma^2;
    I = I + sigma*randn(size(I));

    % Compute moments:
    disp('Estimating sigma')
    
    if isempty(gcp('nocreate')), parpool(maxNumCompThreads); end
    [SNR_est, ~, var_n]=cryo_estimate_snr(cfft2(I)/M); % estimate SNR and noise variance from corners
    sigma = sqrt(var_n)

    % Compute moments:
    disp('Computing moments')
    pad_len = 256 - 2*(L-1);
    [m1_curr, m2_curr, m3_curr] = moments_from_micrograph_steerable(I, L, pad_len);
    
    % Debias moments:
    if SNR < inf
        disp('Debiasing')
        [m2_curr, m3_curr] = debias_moments_steerable(m1_curr, m2_curr, m3_curr,...
            sigma, L);
    end

    m1 = m1 + m1_curr/batch_num;
    m2 = m2 + m2_curr/batch_num;
    m3 = m3 + vec_cell(m3_curr)/batch_num;
end

disp('Saving:')

filename = ['1qlq_crop_' num2str(num_micros) '_M_' num2str(M) '_noisy_moments_SNR_' num2str(SNR) '.mat'];

disp(['Filename: ' filename])

save(filename, 'm1', 'm2', 'm3', 'seed', 'sigma', 'M')

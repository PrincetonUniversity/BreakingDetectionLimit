% September 2018
% https://github.com/PrincetonUniversity/BreakingDetectionLimit
%
% This script conducts the numerical experiments and save the results. 
% The script gen_figure_script.m loads the results and generates the
% figure.
%
% Manopt optimization toolbox is required https://www.manopt.org/

clear;
close all;
clc;

% arbitrary (but fixed) seed
rng(956546);

% use multiple workers if available 
NumWorkers = 2;
if isempty(gcp('nocreate'))
    parpool(NumWorkers,'IdleTimeout', 240);
end

%% Pick parameters and generate signals
% This code can be modified easily for multiple signals. For the purpose of
% this paper, we worked with only one fixed signal. 

% Pick K signals of length L 
K = 1;
L = 21;
W = 2*L-1;

% a "step" signal
X = [ones(ceil(L/2), 1) ; -ones(floor(L/2), 1)];

% pick a noise level
sigma = 3;

% Desired number of occurrences of each signal. 
% We will conduct multiple experiments, one for each elemnt of the vector m_want_vector 
Len_m = 9;
m_want_vector = round(logspace(3, 7, Len_m));

% save all parameters 
save('parameters');

% initialize variables 
error = zeros(Len_m, 1);
gamma = zeros(Len_m, 1);
Xest = zeros(L, Len_m);

%% Conducting one experiment for each of the elements of m_want_vector

for iter = 1:Len_m
     
m_want = m_want_vector(iter); 

% Length of micrograph (the sum is used only for the case of K>1)
n = sum(m_want)*W*10; 
fprintf('Micrograph length: %g\n\n\n', n);

%% Pick which correlation coefficients to sample

[list2, list3] = moment_selection(L, 'exclude biased');

%% Generate the micrograph

T = tic();
% generate micrograph, possiby with K>1 signals
[y_clean, m_actual] = generate_clean_micrograph_1D_heterogeneous(X, W, n, m_want);
y_obs = y_clean + sigma*randn(n, 1);
time_to_generate_micrograph = toc(T);
fprintf('Time to generate micrograph: %.2g [s]\n', time_to_generate_micrograph);
SNR = norm(y_clean, 'fro')/norm(y_obs-y_clean, 'fro');
fprintf('   SNR: %.2g\n', SNR);
fprintf('   m_actual/m_want: ');
fprintf(' %.2g', m_actual./m_want);
fprintf('\n');

%% Collect the moments
T = tic();
batch_size = 1e8;
[M1, M2, M3] = moments_from_data_no_debias_1D_batch(y_obs, list2, list3, batch_size);
time_to_compute_moments = toc(T);
fprintf('   Moment computation: %.4g [s]\n', time_to_compute_moments);

moments.M1 = M1 / n; 
moments.M2 = M2 / n;
moments.M3 = M3 / n;
moments.list2 = list2;
moments.list3 = list3;

clear y_clean y_obs;

%% Optimization

L_optim = 2*L-1;
[Xest(:,iter), gamma(iter), X1, gamma1, X1_L, cost_X2] = heterogeneous_1D(moments, K, L, L_optim, []);
error(iter) = norm(Xest(:,iter) - X(:))/norm(X(:));

save('error','error');
save('Xest','Xest');
fprintf('Relative error = %g\n',error);

fprintf('Estimated densities:\n');
disp(gamma');
fprintf('True densities:\n');
disp(m_actual*L/n);
save('gamma','gamma')

end


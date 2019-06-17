clear;
close all;
clc;

%% Pick parameters and generate signal

% Pick K signals of length L and the size W of the separation window
K = 3;
L = 21; % make this 21
W = 2*L-1;
X = zeros(L, K);
% X(:, 1) = [ones(ceil(L/2), 1) ; -ones(floor(L/2), 1)];
X(:, 1) = circshift([ones(ceil(L/2), 1) ; zeros(floor(L/2), 1)], floor(L/4));
X(:, 2) = [linspace(1, -1, ceil(L/2))' ; linspace(-1, 1, floor(L/2))'];
X(:, 3) = randn(L, 1);

% Pick a noise level
sigma = 3; % make this 3

% Desired number of occurrences of each signal X(:, k)
m_want = [3 2 1]*1e7; % make these multiples of 1e7

% Length of micrograph
n = sum(m_want)*W*5;

fprintf('Micrograph length: %g\n\n\n', n);

%% Pick which correlation coefficients to sample

[list2, list3] = moment_selection(L, 'exclude biased');

%% Generate the micrograph

T = tic();
[y_clean, m_actual] = generate_clean_micrograph_1D_heterogeneous(X, W, n, m_want);
y_obs = y_clean + sigma*randn(n, 1);
time_generate_micrograph = toc(T);
gamma = m_actual*L/n;
SNR = norm(y_clean, 'fro')/norm(y_obs-y_clean, 'fro');

%% The grand experiment starts here

% Select sizes of sub-micrographs to consider.
ns = unique(round(logspace(8, log10(n), 9)));  % first input should be 8

% How many times do we optimize from a different random initial guess?
n_init_optim = 10;

for iter = 1 : length(ns)
    
    result = struct();

    % Collect the moments for the first bit of the micrograph.
    nn = ns(iter);
    T = tic();
    [M1, M2, M3] = moments_from_data_no_debias_1D_batch( ...
                                           y_obs(1:nn), list2, list3, 1e8);    
	%! We normalize by nn here.
    moments.M1 = M1 / nn;
    moments.M2 = M2 / nn;
    moments.M3 = M3 / nn;
    moments.list2 = list2;
    moments.list3 = list3;
    
    result.time_to_compute_moments = toc(T);

    % Parameters and initializations for optimization:
    % empty inputs mean default values are picked.
    L_optim = 2*L-1;
    sigma_est = [];
    X0 = [];
    gamma0 = []; % True value is: m_actual*L_optim/n
    gamma_fixed = false; % Optimize for gamma as well as for the signals
    
    % Run the optimization from different random initializations n_repeat
    % times, and keep the best result according to the cost value of X2.
    result.cost_X2 = inf;
    time_to_optimize = zeros(n_init_optim, 1);
    costs = zeros(n_init_optim, 1);
    for repeat = 1 : n_init_optim
        T = tic();
        [X2, gamma2, X1, gamma1, X1_L, cost_X2] = heterogeneous_1D( ...
               moments, K, L, L_optim, sigma_est, X0, gamma0, gamma_fixed);
        time_to_optimize(repeat) = toc(T);
        costs(repeat) = cost_X2;
        if cost_X2 < result.cost_X2
            result.X1 = X1;
            P1_L = best_permutation(X, X1_L);
            X1_L = X1_L(:, P1_L);
            gamma1 = gamma1(P1_L);
            P2 = best_permutation(X, X2);
            X2 = X2(:, P2);
            gamma2 = gamma2(P2);
            result.X1_L = X1_L;
            result.X2 = X2;
            result.gamma1 = gamma1;
            result.gamma2 = gamma2;
            result.RMSE1 = norm(X1_L-X, 'fro')/norm(X, 'fro');
            result.RMSE2 = norm(X2-X, 'fro')/norm(X, 'fro');
            result.cost_X2 = cost_X2;
        end
    end
    result.costs = costs;
    result.time_to_optimize = time_to_optimize;
    
    % Save in a structure array.
    results(iter) = result; %#ok<SAGROW>
    
    fprintf('\n\n\n ** done %d/%d ** \n\n\n', iter, length(ns));
    
end

clear y_clean y_obs;

ID = randi(1000000);
fprintf('\n\nThis XP ID: %d\n\n', ID);
save(sprintf('heterogeneous_progressive_n%d_%d.mat', n, ID));

%%
% For the experiment where X(:, 1) has zero padding (akin to over-estimated
% support size), we need to cyclically align th eestimator to the ground
% truth, as this cyclic shift is unidentifiable.
for nn = 1 : 9
    X2 = results(nn).X2;
    X2(:, 1) = align_to_reference_1D(X2(:, 1), X(:, 1));
    results(nn).X2 = X2;
end

%%
figure(1);

subplotcounter = 1;
for kk = 1 : K
    snapshots = 1 : 4 : 9;
    for nn = snapshots
        
        subplot(3, 4, subplotcounter);
        result = results(nn);
        X2 = result.X2;
        T = 0:(L-1);
        plot(T, X2(:, kk), T, X(:, kk), 'LineWidth', 1);
        ylim([-2, 2]);
        if kk == 3
            ylim([-2.5, 3.8]);
        end
%         set(gca, 'YTick', [-2, 0, 2]);
        xlim([0, 20]);
        set(gca, 'XTick', [0, 10, 20]);
        set(gca, 'FontSize', 14);
        if nn == 1
            ylabel(sprintf('signal %d', kk));
        end
        subplotcounter = subplotcounter + 1;
        
    end
    
    subplot(3, 4, subplotcounter);
    rmse2 = zeros(size(ns));
    for nn = 1 : length(ns)
        result = results(nn);
        X2 = result.X2;
        rmse2(nn) = norm(X2(:, kk) - X(:, kk), 'fro') / norm(X(:, kk), 'fro');
    end
    loglog(ns, rmse2, '.-', ns(snapshots), rmse2(snapshots), '.');
    xlim([min(ns), max(ns)]);
    ylim([min(1e-2, min(rmse2)), max(1e0, max(rmse2))]);
    set(gca, 'XTick', [1e8, 1e9, 1e10]);
    set(gca, 'YTick', [1e-2, 1e-1, 1e0]);
%     q = polyfit(log10(ns), log10(rmse2), 1);
%     title(sprintf('Relative RMSE; slope: %g', q(1)));
    if kk == 1
        title('Relative RMSE');
    end
    if kk == 3
        xlabel('micrograph length');
    end
    subplotcounter = subplotcounter + 1;
    
end

set(gcf, 'Color', 'w');

%%
figname1 = sprintf('heterogeneous_progressive_n%d_%d', n, ID);
savefig(1, [figname1, '.fig']);
pdf_print_code(1, [figname1 '.pdf'], 13);

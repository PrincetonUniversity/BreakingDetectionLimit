clear;
close all;
clc;

%% Pick parameters and generate signal

% Pick one signal of length L and the size W of the separation window.
L = 21;
W = 2*L-1;
% X = randn(L, 1);
X = zeros(L, 1);
X(1:7) = 1;
X(8:11) = -1;
X(12:L) = .5;
% X = [linspace(0, 2, ceil(L/2))' ; linspace(2, 0, floor(L/2))'];

% Pick a noise level.
sigma = 3;

% Desired length of micrograph.
n = 1e10;

% Desired number of occurrences of the signal.
m_want = floor(n/(5*W));

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
ns = unique(round(logspace(5, log10(n), 9)));

% How many times do we optimize from a different random initial guess?
n_init_optim = 3;

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
    
    % Run the optimization from different random initializations n_repeat
    % times, and keep the best result according to the cost value of X2.
    result.cost_X2 = inf;
    time_to_optimize = zeros(n_init_optim, 1);
    costs = zeros(n_init_optim, 1);
    for repeat = 1 : n_init_optim
        T = tic();
        [X2, gamma2, X1, gamma1, X1_L, cost_X2] = heterogeneous_1D( ...
                            moments, 1, L, L_optim, sigma_est, X0, gamma0);
        time_to_optimize(repeat) = toc(T);
        costs(repeat) = cost_X2;
        if cost_X2 < result.cost_X2
            result.X1 = X1;
            result.X1_L = X1_L;
            result.X2 = X2;
            result.gamma1 = gamma1;
            result.gamma2 = gamma2;
            result.RMSE1 = norm(X1_L-X)/norm(X);
            result.RMSE2 = norm(X2-X)/norm(X);
            result.cost_X2 = cost_X2;
        end
    end
    result.costs = costs;
    result.time_to_optimize = time_to_optimize;
    
    % Save in a structure array.
    results(iter) = result; %#ok<SAGROW>
    
end

clear y_clean y_obs;

ID = randi(1000000);

save(sprintf('progressive_n%d_%d.mat', n, ID));

%%
figure(1);

for iter = 1 : length(ns)
    result = results(iter);
    subplot(3, 3, iter);
    T = 0:(L-1);
    plot(T, result.X1_L, T, result.X2, T, X);
    title(sprintf('n = %d', ns(iter)));
    
    % Hack cosmetics for specific experience
%     if ns(iter) < 2e6
%         ylim([-5, 5]);
%     else
        ylim([-2, 2]);
        set(gca, 'YTick', [-2, 0, 2]);
        set(gca, 'XTick', [0, 10, 20]);
        set(gca, 'FontSize', 14);
%     end
    
%     hleg = legend(sprintf('First estimate (%.2g)', result.gamma1*L/L_optim), ...
%                   sprintf('Final estimate (%.2g)', result.gamma2), ...
%                   sprintf('Ground truth (%.2g)', result.gamma));
%     set(hleg, 'Location', 'northoutside');
%     set(hleg, 'Orientation', 'horizontal');
end

set(gcf, 'Color', 'w');

figname1 = sprintf('progressive_n%d_%d', n, ID);
savefig(1, [figname1, '.fig']);
pdf_print_code(1, [figname1 '.pdf'], 14);

%%
figure(2);

loglog(ns, [results.RMSE1], '.-', ns, [results.RMSE2], '.-');
hleg = legend('Long estimate', 'Final estimate');
set(hleg, 'Location', 'northeast');
set(hleg, 'Orientation', 'vertical');
set(hleg, 'Box', 'off');
title('Root mean squared error on signal estimation');
set(gcf, 'Color', 'w');
xlabel('Observation length n');
ylabel('RMSE');
set(gca, 'FontSize', 14);
grid on;

figname2 = sprintf('progressive_RMSE_n%d_%d', n, ID);
savefig(2, [figname2, '.fig']);
pdf_print_code(2, [figname2 '.pdf'], 14);

P = polyfit(log10(ns), log10([results.RMSE2]), 1);
fprintf('Slope: %g\n', P(1));


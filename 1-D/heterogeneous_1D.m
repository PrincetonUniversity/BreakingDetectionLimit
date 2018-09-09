function [X2, gamma2, X1, gamma1, X1_L, cost_X2] = heterogeneous_1D(moments, K, L, L_optim, sigma_est, X0, gamma0)
% Estimate K signals of length L from moments, with an intermediate
% optimization through signals of length L_optim.
%
% [X2, gamma2, X1, gamma1, X1_L, cost_X2] = heterogeneous_1D(moments, ...
%                                     K, L, L_optim, sigma_est, X0, gamma0)
%
%
% Inputs:
%
% moments: a structure with fields list2, list3, M1, M2, M3 (see examples.)
%
% K >= 1, L >= 1, L_optim >= L integers.
%
% sigma_est (may be empty): estimate for noise standard deviation.
% 
% X0: L_optim x K, initial guess at the signals.
%
% gamma0: K x 1, initial guess at each signal's density in the observation.
%
% 
% Outputs:
%
% X2 (LxK) and gamma2 (Kx1): final estimates for signals and densities.
% X1 (L_optimxK) and gamma1 (Kx1): intermediate (long) estimates.
% X1_L (LxK): regions of interest extracted from X1 to initialize X2.
%

    % Estimate the noise variance (may be inconsequential if biased moments
    % are omitted from the data and if the weights internally do not depend
    % on sigma.)
    if ~exist('sigma_est', 'var') || isempty(sigma_est)
        sigma_est = 0;
    end

    % Optimization is in two stages: we first look for signals of length
    % L_optim, then extract regions of interest of length L and reoptimize.
    if ~exist('L_optim', 'var') || isempty(L_optim)
        L_optim = 2*L-1;
    end

    % Pick an initial guess at the signals with length L_optim.
    if ~exist('X0', 'var') || isempty(X0)
        X0 = randn(L_optim, K);
    end

    % Pick an initial guess at the signal densities.
    if ~exist('gamma0', 'var') || isempty(gamma0)
        gamma0 = .1*ones(K, 1)/K;
    end
    
    % First optimization round.
    [X1, gamma1, problem] = least_squares_1D_heterogeneous(moments, ...
                                     L_optim, K, sigma_est, X0, gamma0(:));

    % Extract best subsignals of length L (with cyclic indexing) in
    % each estimated signal.
    if K > 1
        X1_L = extract_roi(X1, L);
    else
        % Different strategy for K = 1
        % Loss function (getCost on this problem expects an input of length
        % L_long, but we test signals of length L_short, so we zero-pad.)
        lossfun = @(x) getCost(problem, ...
           struct('gamma', gamma1, 'X', [x ; zeros(L_optim - L, 1)]));
        X1_L = extract_roi_loss(X1, L, lossfun);
    end
    
    % Scale estimated densities now that signal length changed.
    gamma2_0 = gamma1*(L/L_optim);

    % Second optimization round.
    [X2, gamma2, ~, ~, cost_X2] = least_squares_1D_heterogeneous(moments, ...
                                       L, K, sigma_est, X1_L, gamma2_0(:));

end

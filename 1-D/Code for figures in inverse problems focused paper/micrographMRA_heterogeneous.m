function result = micrographMRA_heterogeneous(y, sigma, L, K, list2, list3, ...
                                              X0, gamma0, n_init_optim)


    % Collect the moments for the first bit of the micrograph.
    T = tic();
    [M1, M2, M3] = moments_from_data_no_debias_1D_batch(y, list2, list3, 1e8);    
    %! We normalize here.
    moments.M1 = M1 / length(y);
    moments.M2 = M2 / length(y);
    moments.M3 = M3 / length(y);
    moments.list2 = list2;
    moments.list3 = list3;

    result.time_to_compute_moments = toc(T);

    % Parameters and initializations for optimization:
    % empty inputs mean default values are picked.
    L_optim = 2*L-1;
    sigma_est = sigma;
    if ~exist('X0', 'var')
        X0 = [];
    end
    if ~exist('gamma0', 'var')
        gamma0 = []; % True value is: m_actual*L_optim/length(y)
    end
    % How many initializations for the optimization?
    if ~exist('n_init_optim', 'var') || isempty(n_init_optim)
        n_init_optim = 1;
    end

    % Run the optimization from different random initializations n_repeat
    % times, and keep the best result according to the cost value of X2.
    result.cost_X2 = inf;
    time_to_optimize = zeros(n_init_optim, 1);
    costs = zeros(n_init_optim, 1);
    for repeat = 1 : n_init_optim
        T = tic();
        [X2, gamma2, X1, gamma1, X1_L, cost_X2, problem] = heterogeneous_1D( ...
                            moments, K, L, L_optim, sigma_est, X0, gamma0);
        time_to_optimize(repeat) = toc(T);
        costs(repeat) = cost_X2;
        if cost_X2 < result.cost_X2
            result.X1 = X1;
%             P1_L = best_permutation(X, X1_L);
%             X1_L = X1_L(:, P1_L);
%             gamma1 = gamma1(P1_L);
%             P2 = best_permutation(X, X2);
%             X2 = X2(:, P2);
%             gamma2 = gamma2(P2);
            result.X1_L = X1_L;
            result.X2 = X2;
            result.gamma1 = gamma1;
            result.gamma2 = gamma2;
%             result.RMSE1 = norm(X1_L-X, 'fro')/norm(X, 'fro');
%             result.RMSE2 = norm(X2-X, 'fro')/norm(X, 'fro');
            result.cost_X2 = cost_X2;
            result.problem = problem;
        end
    end
    result.costs = costs;
    result.time_to_optimize = time_to_optimize;
    
end

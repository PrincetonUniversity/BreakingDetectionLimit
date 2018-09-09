function [x_est, m_est, problem, stats] = least_squares_1D_params(M1, M2, M3, W, sigma, N, L, list2, list3, x0, m0)

    params.W = W;
    params.N = N;
    params.M1 = M1;
    params.M2 = M2;
    params.M3 = M3;
    params.list2 = list2;
    params.list3 = list3;
    params.sigma = sigma;

    %% Precompute biases once and for all
    n2 = size(list2, 1);
    bias2 = zeros(n2, 1);
    for k = 1 : n2
        shift = list2(k);
        if shift == 0
            bias2(k) = N*sigma^2;
        end
    end
    
    n3 = size(list3, 1);
    bias3 = zeros(n3, 1);
    for k = 1 : n3
        shift1 = list3(k, 1);
        shift2 = list3(k, 2);
        if shift1 == 0
            bias3(k) = bias3(k) + M1*sigma^2; % this is the 'data' M1 (fixed)
        end
        if shift2 == 0
            bias3(k) = bias3(k) + M1*sigma^2;
        end
        if shift1 == shift2
            bias3(k) = bias3(k) + M1*sigma^2;
        end
    end
    
    params.bias2 = sparse(bias2);
    params.bias3 = sparse(bias3);

    %% Setup Manopt problem

    % Choose wisely if optimize in R^L or R^W.
    % It seems that W might be necessary to avoid local optima.
    elements.x = euclideanfactory(W, 1);
    elements.m_var = euclideanfactory(1, 1);
    manifold = productmanifold(elements);
    
    problem.M = manifold;
    problem.costgrad = @(X) least_squares_1D_cost_grad_params_parallel(X, params);

    if ~exist('x0', 'var') || isempty(x0)
        x0 = elements.x.rand();
    end
    if ~exist('m0', 'var') || isempty(m0)
        m0 = randi(round(N/L));
    end
    X0.x = x0;
    X0.m_var = m0;
    
    opts = struct();
%    opts.tolgradnorm = 1e-5;
    opts.tolgradnorm = 1e-3;

    opts.maxiter = 1000;
    warning('off', 'manopt:getHessian:approx');
    [X_est, loss, stats] = trustregions(problem, X0, opts); %#ok<ASGLU>
    x_est = X_est.x;
    m_est = X_est.m_var;
    warning('on', 'manopt:getHessian:approx');

end

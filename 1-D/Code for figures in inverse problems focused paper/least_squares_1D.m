function [x_est, problem, stats] = least_squares_1D(M1, M2, M3, W, sigma, N, L, m, list2, list3, x0) %#ok<INUSL>

    params.W = W;
    params.N = N;
    params.m = m;
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
    manifold = euclideanfactory(W, 1);
    
    problem.M = manifold;
    problem.costgrad = @(x) least_squares_1D_cost_grad_parallel(x, params);

    if ~exist('x0', 'var')
        x0 = [];
    end

%     opts = struct();
%     opts.maxiter = 500;
%     [x_est, loss] = rlbfgs(problem, x0, opts); %#ok<ASGLU>
%     x0 = x_est;
    
    opts = struct();
    opts.tolgradnorm = 1e-5;
    opts.maxiter = 1000;
    warning('off', 'manopt:getHessian:approx');
    [x_est, loss, stats] = trustregions(problem, x0, opts); %#ok<ASGLU>
    warning('on', 'manopt:getHessian:approx');

end

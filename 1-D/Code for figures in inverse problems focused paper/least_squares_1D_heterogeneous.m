function [X_est, gamma_est, problem, stats, loss] = least_squares_1D_heterogeneous(moments, L_optim, K, sigma, X0, gamma0)

    M1 = moments.M1;        %% Make sure these are normalized by n (length of the micrograph)
    M2 = moments.M2;
    M3 = moments.M3;
    list2 = moments.list2;
    list3 = moments.list3;

    params.M1 = M1;
    params.M2 = M2;
    params.M3 = M3;
    params.list2 = list2;
    params.list3 = list3;
    params.sigma = sigma;
    
    %%%%%%%%
    %%
    %%
    %%    The data is now normalized by N: change everywhere, be careful, inform Tamir. This is a better match with the math / better for scaling.
    %%    Also clarify roles of N, L, W: this function doesn't need to know about W: should just get L, and outer call pretends x's have length L when really it's L/2, that's all.
    %%
    %%%%%%%%

    
    %% Precompute biases once and for all
    
    n2 = size(list2, 1);
    bias2 = zeros(n2, 1);
    for k = 1 : n2
        shift = list2(k);
        if shift == 0
            bias2(k) = sigma^2;
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
    elements.X = euclideanfactory(L_optim, K);
    
%     elements.gamma = euclideanfactory(K, 1);  %% TODO: try unrestricted reals
%     elements.gamma = positivefactory(K, 1);  %% TODO: try positive reals factory
    elements.gamma = constantfactory(gamma0);  %% TODO: try fixing gamma; requires gamma0
    fprintf('\n\n ### GAMMA IS FIXED ###\n\n\n');

    manifold = productmanifold(elements);
    
    problem.M = manifold;
    
    problem.costgrad = @costgrad;
    function [f, G] = costgrad(Z)
        [f, G] = least_squares_1D_cost_grad_heterogeneous(Z, params);
        G = manifold.egrad2rgrad(Z, G);
    end
    
    % checkgradient(problem); pause;

    %% Pick an initial guess if not provided
    if ~exist('X0', 'var')
        X0 = randn(L_optim, K);
    end
    if ~exist('gamma0', 'var')
        gamma0 = 0.1 * (ones(K, 1) / K);
    end
    Z0.X = X0;
    Z0.gamma = gamma0;
    
    %% Call an optimization algorithm
    opts = struct();
    opts.tolgradnorm = 1e-10; %1e-12;
    opts.maxiter = 1000;
    
    fprintf(' ## MAX TIME ##\n');
    opts.maxtime = 30*60;
    
    warning('off', 'manopt:getHessian:approx');
    [Z, loss, stats] = trustregions(problem, Z0, opts);
    warning('on', 'manopt:getHessian:approx');
    
    X_est = Z.X;
    gamma_est = Z.gamma;

end

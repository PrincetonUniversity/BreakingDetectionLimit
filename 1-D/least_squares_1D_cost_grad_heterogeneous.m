function [f, G] = least_squares_1D_cost_grad_heterogeneous(Z, params, sample3)

    M1data = params.M1;
    M2data = params.M2;
    M3data = params.M3;
    list2 = params.list2;
    list3 = params.list3;
    bias2 = params.bias2;
    bias3 = params.bias3;
    
    % sample3 can be used to compute the cost and gradient only with
    % respect to a subsample of the 3rd order moments in list3.
    % This is used for Manopt's stochasticgradient solver.
    if ~exist('sample3', 'var') || isempty(sample3)
        sample3 = 1 : size(list3, 1);
    end
    sample3 = sample3(:); % make sure it is a column
    
    
    % We optimize for both the signals and for mixing parameters.
    X = Z.X;
    gamma = Z.gamma;
    
    % This L is as defined by the size of the optimization variable: it
    % need not be equal to the L as defined by the size of the true signal.
    % K is the heterogeneity parameter: there are K signals of length L.
    [L_optim, K] = size(X);
    assert(all(size(gamma) == [K, 1]), 'Requirement for fields of Z: X has size L_optim x K and gamma has size Kx1');
    
    need_gradient = (nargout() >= 2);
    
    % TODO: change with the weights from https://arxiv.org/abs/1710.02590
    w1 = 1;
    w2 = 1/size(list2, 1);
    w3 = 1/size(sample3, 1);
    
    % First-order moment, forward model
    mean_X = sum(X, 1)'/L_optim;
    M1 = gamma'*mean_X;
    R1 = M1 - M1data;
    f = .5*w1*R1^2;
    
    if need_gradient
        g_X = (w1*R1/L_optim)*ones(L_optim, 1)*gamma';
        g_gamma = w1*R1*mean_X;
    end
    
    
    
    % Second-order moments, forward model
    n2 = size(list2, 1);
    for element = 1 : n2
        
        shift1 = list2(element);
        
        vals1 = [0, -shift1];
        range1 = ((1+max(vals1)) : (L_optim+min(vals1)))';
        X1 = X(range1, :);
        X2 = X(range1+shift1, :);
        
        mean_X1_X2 = sum(X1 .* X2, 1)'/L_optim;
        M2k = gamma'*mean_X1_X2 + bias2(element);
        
        R2k = M2k - M2data(element);
        f = f + .5*w2*R2k^2;
        
        % This part of the code for gradient only
        if need_gradient
            T1 = zeros(L_optim, K);
            T1(range1, :) = X2;
            T2 = zeros(L_optim, K);
            T2(range1+shift1, :) = X1;
            T = T1 + T2;
            T = bsxfun(@times, T, gamma'); % scale column k by gamma(k)
            g_X = g_X + (w2*R2k/L_optim)*T;
            g_gamma = g_gamma + w2*R2k*mean_X1_X2;
        end
        
    end
    
    
    
    % Third-order moments, forward model
    sample3 = sample3(:);
    n3 = size(sample3, 1);
    for sample = 1 : n3
        
        element = sample3(sample);

        shifts = list3(element, :);
        shift1 = shifts(1);
        shift2 = shifts(2);

        vals1 = [0, -shift1, -shift2];
        range1 = ((1+max(vals1)) : (L_optim+min(vals1)))';
        X1 = X(range1, :);
        X2 = X(range1+shift1, :);
        X3 = X(range1+shift2, :);
        X1X2 = X1 .* X2;

        mean_X1_X2_X3 = sum(X1X2 .* X3, 1)' / L_optim;
        M3k = gamma'*mean_X1_X2_X3 + bias3(element);

        R3k = M3k - M3data(element);
        f = f + .5*w3*R3k^2;

        % This part of the code for gradient only
        if need_gradient
            T1 = zeros(L_optim, K);
            T1(range1, :) = X2.*X3;
            T2 = zeros(L_optim, K);
            T2(range1+shift1, :) = X1.*X3;
            T3 = zeros(L_optim, K);
            T3(range1+shift2, :) = X1X2;
            T = T1 + T2 + T3;
            T = bsxfun(@times, T, gamma'); % scale column k by gamma(k)
            g_X = g_X + (w3*R3k/L_optim)*T;
            g_gamma = g_gamma + w3*R3k*mean_X1_X2_X3;
        end
        
    end

    
    %% Setup a structure for the gradient
    if need_gradient
        G = struct();
        G.X = g_X;
        G.gamma = g_gamma;
    end
    
end

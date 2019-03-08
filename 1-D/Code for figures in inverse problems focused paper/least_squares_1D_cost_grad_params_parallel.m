function [f, G] = least_squares_1D_cost_grad_params_parallel(X, params, sample3)

    N = params.N; %#ok<NASGU>
    W = params.W;
    M1data = params.M1;
    M2data = params.M2;
    M3data = params.M3;
    sigma = params.sigma; %#ok<NASGU>
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
    
    
    % We optimize for both the signal and for parameters
    x = X.x;
    m_var = X.m_var; % make it m_var ? would be easier...
    
    
    % This L is as defined by the size of the optimization variable: it
    % need not be equal to the L as defined by the size of the true signal.
    x = x(:);
    LL = size(x, 1);
    
    need_gradient = (nargout() >= 2);
    
    % TODO: change with the weights from https://arxiv.org/abs/1710.02590
    w1 = 1;
    w2 = 1/W;
    w3 = 1/(W^2);
    
    % First-order moment, forward model
    sum_x = sum(x(:));
    M1 = m_var*sum_x;
    R1 = M1 - M1data;
    f = .5*w1*R1^2;
    
    if need_gradient
        g = m_var*(w1*R1)*ones(LL, 1);
        g_m = w1*R1*sum_x;
    end
    
    
    
    % Second-order moments, forward model
    n2 = size(list2, 1);
    for k = 1 : n2
        
        shift1 = list2(k);
        
        vals1 = [0, shift1];
        range1 = (1+max(vals1)) : (LL+min(vals1));
        x1 = x(range1);
        x2 = x(range1-shift1(1));
        
        sum_x1_x2 = sum(x1 .* x2);
        M2k = m_var*sum_x1_x2 + bias2(k);
        
        R2k = M2k - M2data(k);
        f = f + .5*w2*R2k^2;
        
        % This part of the code for gradient only
        if need_gradient
            T1 = zeros(LL, 1);
            T1(range1) = x2;
            T2 = zeros(LL, 1);
            T2(range1-shift1(1)) = x1;
            T = T1 + T2;
            g = g + m_var*w2*R2k*T;
            g_m = g_m + w2*R2k*sum_x1_x2;
        end
        
    end
    
    
    % Third-order moments, forward model
    sample3 = sample3(:); % make sure it's a column vector
    sn3 = size(sample3, 1);
    
    ppool = gcp('nocreate');
    numworkers = min(sn3, ppool.NumWorkers);
    separators = round(linspace(1, sn3+1, numworkers+1));
    limits = zeros(numworkers, 2);
    for worker = 1 : numworkers
        limits(worker, :) = [separators(worker), separators(worker+1)-1];
    end
    
    fs = zeros(numworkers, 1);
    Gs = zeros(LL, numworkers);
    Gs_m = zeros(1, numworkers);
    
    parfor worker = 1 : numworkers
        
        ff = 0;
        GG = zeros(LL, 1);
        gg_m = 0;
        these_limits = limits(worker, :);
        for kk = these_limits(1) : these_limits(2)
            
            k = sample3(kk);

            shifts = list3(k, :);
            shift1 = shifts(1);
            shift2 = shifts(2);

            vals1 = [0, shift1, shift2];
            range1 = (1+max(vals1)) : (LL+min(vals1));
            x1 = x(range1);
            x2 = x(range1-shift1);
            x3 = x(range1-shift2);
            x1x2 = x1 .* x2;

            sum_x1_x2_x3 = sum(x1x2 .* x3);
            M3k = m_var*sum_x1_x2_x3 + bias3(k);

            R3k = M3k - M3data(k);
            ff = ff + .5*w3*R3k^2;

            % This part of the code for gradient only
            if need_gradient
                T1 = zeros(LL, 1);
                T1(range1) = x2.*x3;
                T2 = zeros(LL, 1);
                T2(range1-shift1) = x1.*x3;
                T3 = zeros(LL, 1);
                T3(range1-shift2) = x1x2;
                T = T1 + T2 + T3;
                GG = GG + m_var*w3*R3k*T;
                gg_m = gg_m + w3*R3k*sum_x1_x2_x3;
            end
            
        end
        
        fs(worker) = ff;
        if need_gradient
            Gs(:, worker) = GG;
            Gs_m(worker) = gg_m;
        end
        
    end
    
    f = f + sum(fs);
    if need_gradient
        g = g + sum(Gs, 2);
        g_m = g_m + sum(Gs_m);
    end
        
    if need_gradient
        G = struct();
        G.x = g;
        G.m_var = g_m;
    end
    
    
end

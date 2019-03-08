function [f, g] = least_squares_1D_cost_grad_parallel(x, params, sample3)

    N = params.N; %#ok<NASGU>
    m = params.m;
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
    
    % This L is as defined by the size of the optimization variable: it
    % need not be equal to the L as defined by the size of the true signal.
    x = x(:);
    L = size(x, 1);
    
    need_gradient = (nargout() >= 2);
    
    % TODO: change with the weights from https://arxiv.org/abs/1710.02590
    w1 = 1;
    w2 = 1/W;
    w3 = 1/(W^2);
    
    % First-order moment, forward model
    M1 = m*sum(x(:));
    R1 = M1 - M1data;
    f = .5*w1*R1^2/m;
    
    if need_gradient
        g = (w1*R1)*ones(L, 1);
    end
    
    
    
    % Second-order moments, forward model
    n2 = size(list2, 1);
    for k = 1 : n2
        
        shift1 = list2(k);
        
        vals1 = [0, shift1];
        range1 = (1+max(vals1)) : (L+min(vals1));
        x1 = x(range1);
        x2 = x(range1-shift1(1));
        
        M2k = m*sum(x1 .* x2) + bias2(k);
        
        R2k = M2k - M2data(k);
        f = f + .5*w2*R2k^2/m;
        
        % This part of the code for gradient only
        if need_gradient
            T1 = zeros(L, 1);
            T1(range1) = x2;
            T2 = zeros(L, 1);
            T2(range1-shift1(1)) = x1;
            T = T1 + T2;
            g = g + w2*R2k*T;
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
    Gs = zeros(L, numworkers);
    
    parfor worker = 1 : numworkers
        
        ff = 0;
        GG = zeros(L, 1);
        these_limits = limits(worker, :);
        for kk = these_limits(1) : these_limits(2)
            
            k = sample3(kk);

            shifts = list3(k, :);
            shift1 = shifts(1);
            shift2 = shifts(2);

            vals1 = [0, shift1, shift2];
            range1 = (1+max(vals1)) : (L+min(vals1));
            x1 = x(range1);
            x2 = x(range1-shift1);
            x3 = x(range1-shift2);
            x1x2 = x1 .* x2;

            M3k = m*sum(x1x2 .* x3) + bias3(k);

            R3k = M3k - M3data(k);
            ff = ff + .5*w3*R3k^2/m;

            % This part of the code for gradient only
            if need_gradient
                T1 = zeros(L, 1);
                T1(range1) = x2.*x3;
                T2 = zeros(L, 1);
                T2(range1-shift1) = x1.*x3;
                T3 = zeros(L, 1);
                T3(range1-shift2) = x1x2;
                T = T1 + T2 + T3;
                GG = GG + w3*R3k*T;
            end
            
        end
        
        fs(worker) = ff;
        if need_gradient
            Gs(:, worker) = GG;
        end
        
    end
    
    f = f + sum(fs);
    if need_gradient
        g = g + sum(Gs, 2);
    end
    
end

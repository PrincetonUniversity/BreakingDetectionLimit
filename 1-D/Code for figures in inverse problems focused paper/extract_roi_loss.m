function X_short = extract_roi_loss(X_long, L_short, lossfun)
% Extract region of interest of 1 signal based on a loss function
%
% function X_short = extract_roi(X_long, L_short, lossfun)
%
% Given 1 signal of length L_long as a column vector X_long, extracts a
% contiguous subsignal of length L_short with best loss (the region of
% interest), and returns it as the column vector X_short. Loss is computed
% from the given function handle lossfun, which takes as input a signal of
% length L_short.

    [L_long, K] = size(X_long);
    assert(K == 1, 'Implemented only for K = 1 for now.');
    assert(L_long >= L_short, 'Can only extract a shorter region.');
    
    % Save here best subsignal of length L_short
    X_short = zeros(L_short, 1);
    loss = inf;

    for s = 0 : (L_long - L_short)
        x = X_long(s+(1:L_short));
        lossx = lossfun(x);
        if lossx < loss
            X_short = x;
            loss = lossx;
        end
    end

end

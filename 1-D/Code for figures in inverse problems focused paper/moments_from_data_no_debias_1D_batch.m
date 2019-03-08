function [M1, M2, M3] = moments_from_data_no_debias_1D_batch(y, list2, list3, batch_size)
% y is a signal (a vector)
% list2 has size n2 x 1
% list3 has size n3 x 2
% Each row of list2 and list3 contains integers (shifts) -- can be negative
% batch_size: moments are computed on subsignals of y of length batch_size
% at most, then agregated. This is not exact because of missing information
% at the junction points, but is negligible if batch_size is on the order
% of millions or more. This helps to compute on GPUs which have limited
% memory. It is exact if batch_size >= numel(y).
% This code assumes moments_from_data_no_debias_1D returns un-normalized
% moments (not divided by the length of y) and does the same.

    y = y(:);
    n = size(y, 1);
    N = ceil(n / batch_size);
    q = round(linspace(1, n+1, N+1));
    
    n2 = size(list2, 1);
    n3 = size(list3, 1);
    
    M1 = 0;
    M2 = zeros(n2, 1);
    M3 = zeros(n3, 1);
    for k = 1 : N
        [M1add, M2add, M3add] = moments_from_data_no_debias_1D(y(q(k):(q(k+1)-1)), list2, list3);
        M1 = M1 + M1add;
        M2 = M2 + M2add;
        M3 = M3 + M3add;
    end
    
end

function [M1, M2, M3] = moments_from_data_no_debias_1D(y, list2, list3)
% y is a signal (a vector)
% list2 has size n2 x 1
% list3 has size n3 x 2
% Each row of list2 and list3 contains integers (shifts) -- can be negative

    y = y(:);
    n = size(y, 1);
    
    gpuFlag = (gpuDeviceCount > 0);
%     gpuFlag = false;
    
    n2 = size(list2, 1);
    n3 = size(list3, 1);
    assert(size(list2, 2) == 1, 'list2 must have size n2 x 1.');
    assert(size(list3, 2) == 2, 'list3 must have size n3 x 2.');
    
    if gpuFlag
        y = gpuArray(y);
        M2 = zeros(n2, 1, 'gpuArray');
        M3 = zeros(n3, 1, 'gpuArray');
    else
        M2 = zeros(n2, 1);
        M3 = zeros(n3, 1);
    end

    M1 = sum(y);
    
    for k = 1 : n2
        
        shift1 = list2(k);
        vals1 = [0, -shift1];
        
%         M2(k) = 0;
%         for q = (1+max(vals1)) : (n+min(vals1))
%             M2(k) = M2(k) + y(q)*y(q+shift1);
%         end
        
        range1 = (1+max(vals1)) : (n+min(vals1));
        M2(k) = sum(y(range1) .* y(range1+shift1));
        
    end
    
    for k = 1 : n3  % parfor here and no loop inside seems to be fastest on latte (not on laptop; should try GPU code) // parfor creates communication issues one Latte/polar though -- try without, and reassess use of for-loop within
        
        shifts = list3(k, :);
        shift1 = shifts(1);
        shift2 = shifts(2);
        vals1 = [0, -shift1, -shift2];

%         M3(k) = 0;
%         for q = (1+max(vals1)) : (n+min(vals1))
%             M3(k) = M3(k) + y(q)*y(q+shift1)*y(q+shift2);
%         end
        
        range1 = (1+max(vals1)) : (n+min(vals1));
        M3(k) = sum(y(range1) .* y(range1+shift1) .* y(range1+shift2));
        
    end
    
    if gpuFlag
        M1 = gather(M1);
        M2 = gather(M2);
        M3 = gather(M3);
    end
    
end

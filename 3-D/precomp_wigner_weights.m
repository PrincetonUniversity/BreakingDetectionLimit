function W = precomp_wigner_weights(maxL)
% Precompute weights needed for bispectrum evaluation, proportional to
% Wigner 3j symbols.
% 
% Inputs:
%   * maxL: cutoff for spherical harmonics expansion
% 
% Outputs:
%   * W: precompute weights
% 
% Eitan Levin, June 2018

W = cell(maxL+1, maxL+1);
for ll = 1:(maxL+1)^2
    [L1, L2] = ind2sub([maxL+1, maxL+1], ll);
    L1 = L1-1; L2 = L2-1;
    
    L3_vals = abs(L1-L2):min(L1+L2, maxL);
    W{ll} = zeros(2*L2+1, 2*L1+1, length(L3_vals));
    for ii_3 = 1:length(L3_vals)
        L3 = L3_vals(ii_3);
        
        for m1 = -L1:L1
            for m2 = -L2:L2
                m3 = m1+m2;
                if abs(m3) > L3, continue; end
                W{ll}(m2+L2+1, m1+L1+1, ii_3) = (-1)^(m3)*wigner3j(L1, L2, L3, m1, m2, -m3);
            end
        end
    end
end
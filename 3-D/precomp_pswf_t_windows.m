function [Wt, ang_freq] = precomp_pswf_t_windows(L, beta, T)
% Function to compute PSWF windows to compute correlations with micrographs
% 
% Inputs:
%   * L: length of volume or projection
%   * beta: fraction of Nyquist to assume as the bandlimit
%   * T: truncation parameter controlling length of PSWF expansion
%   
% Outputs:
%   * Wt: stack of windows
%   * ang_freq: list of angular frequencies corresponding to each row of Wt
% 
% Eitan Levin, August 2018

[Mt, ang_freq] = precomp_pswf_t_mat_direct(L, beta, T);
Mt = Mt(sum(ang_freq > 0)+1:end, :);

Lr = floor(L/2);
if mod(L, 2) > 0
    [x,y] = meshgrid(-Lr:Lr, -Lr:Lr);
else
    [x,y] = meshgrid(-Lr:Lr-1, -Lr:Lr-1);
end

pts_notin_disc = sqrt(x.^2 + y.^2) > L;
Mt(:, pts_notin_disc) = 0;
Wt = reshape(Mt, size(Mt,1), L, L);
Wt = permute(Wt, [2,3,1]);
Wt = Wt(end:-1:1, end:-1:1, :);

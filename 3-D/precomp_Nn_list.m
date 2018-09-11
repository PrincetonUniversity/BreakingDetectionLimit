function n_list = precomp_Nn_list(L, beta, T)
% Compute list of radial frequencies for PSWFs
% 
% Inputs:
%   * L: length of volume or projection
%   * beta: fraction of Nyquist to assume as the bandlimit
%   * T: truncation parameter for PSWF expansion truncation
% 
% Outputs:
%   * n_list: list of number of radial frequencies for each angular
%   frequency
% 
% Eitan Levin, June 2018

realFlag = true;
PSWF_Nn_p = precomp_pswf_t(L, beta, T, realFlag);
ang_freq = PSWF_Nn_p.ang_freq;
n_list = zeros(max(ang_freq)+1, 1);
for N = 0:max(ang_freq)
    n_list(N+1) = length(find(ang_freq == N));
end

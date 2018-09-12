function M2_micrograph = computeM2(Y_obs,m_eff,param)

% Given a batch of micrographs, compute their averaged second-moment

% Input:
% Y_obs : a batch of noisy micrographs
% m_eff : a vector of number of image repetitions in each micrograph
% param : the parameters of the problem

% Output:
% M2_micrograph : the averaged power spectrum of the micrographs

% September 2018
% https://github.com/PrincetonUniversity/BreakingDetectionLimit

sigma = param.sigma;
L = param.L;
micrograph_batch = param.micrograph_batch;
N = param.N;
W = param.W;

% computing the second moment of the micropgrah
AC = ifft2(abs(fft2(Y_obs)).^2);
M2_micrograph = zeros(2*L-1,2*L-1,micrograph_batch);
M2_micrograph(1:L,1:L,:) = AC(1:L,1:L,:);

% enforcing symmetry
M2_micrograph(L+1:2*L-1,1:L,:) = AC(N-L+2:N,1:L,:);
M2_micrograph(1:L,L+1:2*L-1,:) = AC(1:L,N-L+2:N,:);
M2_micrograph(L+1:2*L-1,L+1:2*L-1,:) = AC(N-L+2:N,N-L+2:N,:);

% debias
M2_micrograph(1,1,:) = M2_micrograph(1,1,:) - N^2*sigma^2;

% normalization
Normalize_factor = repmat(m_eff,[1,W,W]);
Normalize_factor = permute(Normalize_factor,[2,3,1]);
M2_micrograph = M2_micrograph./Normalize_factor;
M2_micrograph = sum(M2_micrograph,3)./micrograph_batch;

end
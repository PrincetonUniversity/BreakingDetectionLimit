function [M2_micrograph, m] = computeM2(Y_obs,m_eff,param)

sigma = param.sigma;
L = param.L;
micrograph_batch = param.micrograph_batch;
N = param.N;
W = param.W;

% computing the second moment of the micropgrah
AC = ifft2(abs(fft2(Y_obs)).^2); %./m_eff;
M2_micrograph = zeros(2*L-1,2*L-1,micrograph_batch);
M2_micrograph(1:L,1:L,:) = AC(1:L,1:L,:);
M2_micrograph(L+1:2*L-1,1:L,:) = AC(N-L+2:N,1:L,:);
M2_micrograph(1:L,L+1:2*L-1,:) = AC(1:L,N-L+2:N,:);
M2_micrograph(L+1:2*L-1,L+1:2*L-1,:) = AC(N-L+2:N,N-L+2:N,:);
M2_micrograph(1,1,:) = M2_micrograph(1,1,:) - N^2*sigma^2;
Normalize_factor = repmat(m_eff,[1,W,W]);
Normalize_factor = permute(Normalize_factor,[2,3,1]);
M2_micrograph = M2_micrograph./Normalize_factor;
M2_micrograph = sum(M2_micrograph,3)./micrograph_batch;

end
function [X_out, discrepancy_norm,err,err1,err2] = RRR(Y,X_init,X_true,param)

% Given an estimation of the image's power spectrum, the RRR aims to find
% the underlying image under the assumption that the image has nonzero
% values only in the upper-left corner and the values of the image are
% within [-1,1].
% The algorithm is described is detail in:
% 1.  Elser, Veit. "Matrix product constraints by projection methods."
% Journal of Global Optimization 68.2 (2017): 329-355.
% 2. Elser, Veit, Ti-Yen Lan, and Tamir Bendory. "Benchmark problems for phase retrieval."
% arXiv preprint arXiv:1706.00399 (2017).

% Input:
% Y : the estimated Fourier magnitudes of the image (the square root of the
% estimated power spectrum).
% X_init : initial guess
% X_true : the true image. Used only to measure error
% param : the parameters of the problem

% Output:
% X_out : the estimated image
% discrepancy_norm: the norm of the discrepancy at each iteration. Used to
% halt the iterations
% err1 : error compared to the ground truth
% err2 : error compared to the ground truth after reflection
% err : min (err1,err2)

% September 2018
% https://github.com/PrincetonUniversity/BreakingDetectionLimit

W = param.W;
L = param.L;
th = param.th;
max_iter = param.max_iter;

%% Projections

% P1 - support projection
Mask = zeros(W);
Mask(1:L,1:L) = 1;
P1 = @(X) Mask.*X;

% P2 - Fourier magnitude projection
P2 = @(Z) Y.*sign(Z);

% P3 - bounds
P3 = @(Z) (abs(Z)>1).*sign(Z) + (abs(Z)<=1).*Z;

%% Parameters
% constant RRR parameter
beta = 1;

% defaults
if ~exist('X_init','var')
    X_init = zeros(W); X_init(1:L,1:L) = rand(L);
end

if ~exist('th','var')
    th = 0;
end

if ~exist('max_iter','var')
    max_iter = 500;
end

%% Initializations

discrepancy_norm = zeros(max_iter,1);
err = zeros(max_iter,1);
err1 = zeros(max_iter,1);
err2 = zeros(max_iter,1);
X = X_init;

%% RRR iterations

for k = 1:max_iter
    
    X1 = P1(X);
    X1 = P3(X1);
    
    % compute error to ground truth to asses progress
    err1(k) = norm(X1(1:L,1:L) - X_true,'fro')/norm(X_true(:));
    err2(k) = norm(rot90(X1(1:L,1:L),2) - X_true,'fro')/norm(X_true(:));
    err(k) = min(err1(k),err2(k));
    
    X2 = real(ifft2(P2(fft2(2*X1 - X))));
    
    discrepancy = X2 - X1;
    X = X + beta*discrepancy;
    
    % Stopping criterion
    discrepancy_norm(k) = norm(discrepancy,'fro')/norm(X(:));
    if discrepancy_norm(k)<th
        fprintf('RRR last iteration = %d\n',k);
        break;
    end
end

%% Final estimate

X = P1(X);
X = P3(X);
err1(k) = norm(X(1:L,1:L) - X_true,'fro')/norm(X_true(:));
err2(k) = norm(rot90(X(1:L,1:L),2) - X_true,'fro')/norm(X_true(:));
err(k) = min(err1(k),err2(k));
if err(k) == err1(k)
    X_out = X;
else
    X_out = zeros(W);
    X_out(1:L,1:L) = rot90(X(1:L,1:L),2);
end

if k == max_iter
    fprintf('RRR last iteration = %d\n',max_iter);
end
discrepancy_norm = discrepancy_norm(1:k);
err = err(1:k);

end


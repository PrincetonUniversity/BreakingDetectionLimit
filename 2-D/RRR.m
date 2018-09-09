function [X_out, discrepancy_norm,err,err1,err2] = RRR(Y,X_init,X_true,param)

W = param.W;
L = param.L;
th = param.th;
max_iter = param.max_iter;

% P1 - support projection
Mask = zeros(W);
Mask(1:L,1:L) = 1;
P1 = @(X) Mask.*X;
% P2 - Fourier magnitude projection
P2 = @(Z) Y.*sign(Z);
% P3 - bounds
P3 = @(Z) (abs(Z)>1).*sign(Z) + (abs(Z)<=1).*Z;
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

discrepancy_norm = zeros(max_iter,1);
err = discrepancy_norm;
err1 = err;
err2 = err;
X = X_init;
for k = 1:max_iter
    X1 = P1(X);
    X1 = P3(X1);
    err1(k) = norm(X1(1:L,1:L) - X_true,'fro')/norm(X_true(:));
    err2(k) = norm(rot90(X1(1:L,1:L),2) - X_true,'fro')/norm(X_true(:));
    err(k) = min(err1(k),err2(k));
    X2 = real(ifft2(P2(fft2(2*X1 - X))));
    discrepancy = X2 - X1;
    X = X + beta*discrepancy;
    discrepancy_norm(k) = norm(discrepancy,'fro')/norm(X(:));
        if discrepancy_norm(k)<th
        fprintf('RRR last iteration = %d\n',k);
        break;
    end
end

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


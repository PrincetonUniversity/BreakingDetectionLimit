function x = discretesample(p, n)
% Samples from a discrete distribution
%
%   x = discretesample(p, n)
%       independently draws n samples (with replacement) from the 
%       distribution specified by p, where p is a probability array 
%       whose elements sum to 1.
%
%       Suppose the sample space comprises K distinct objects, then
%       p should be an array with K elements. In the output, x(i) = k
%       means that the k-th object is drawn at the i-th trial.
%       
%   Remarks
%   -------
%       - This function is mainly for efficient sampling in non-uniform 
%         distribution, which can be either parametric or non-parametric.         
%
%       - The function is implemented based on histc, which has been 
%         highly optimized by mathworks. The basic idea is to divide
%         the range [0, 1] into K bins, with the length of each bin 
%         proportional to the probability mass. And then, n values are
%         drawn from a uniform distribution in [0, 1], and the bins that
%         these values fall into are picked as results.
%
%       - This function can also be employed for continuous distribution
%         in 1D/2D dimensional space, where the distribution can be
%         effectively discretized.
%
%       - This function can also be useful for sampling from distributions
%         which can be considered as weighted sum of "modes". 
%         In this type of applications, you can first randomly choose 
%         a mode, and then sample from that mode. The process of choosing
%         a mode according to the weights can be accomplished with this
%         function.
%
%   Examples
%   --------
%       % sample from a uniform distribution for K objects.
%       p = ones(1, K) / K;
%       x = discretesample(p, n);
%
%       % sample from a non-uniform distribution given by user
%       x = discretesample([0.6 0.3 0.1], n);
%
%       % sample from a parametric discrete distribution with
%       % probability mass function given by f.
%       p = f(1:K);
%       x = discretesample(p, n);
%

% Copyright (c) 2009, Dahua Lin
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.


%   Created by Dahua Lin, On Oct 27, 2008
%

%% parse and verify input arguments

assert(isfloat(p), 'discretesample:invalidarg', ...
    'p should be an array with floating-point value type.');

assert(isnumeric(n) && isscalar(n) && n >= 0 && n == fix(n), ...
    'discretesample:invalidarg', ...
    'n should be a nonnegative integer scalar.');

%% main

% process p if necessary

K = numel(p);
if ~isequal(size(p), [1, K])
    p = reshape(p, [1, K]);
end

% construct the bins

edges = [0, cumsum(p)];
s = edges(end);
if abs(s - 1) > eps
    edges = edges * (1 / s);
end

% draw bins

rv = rand(1, n);
c = histc(rv, edges);
ce = c(end);
c = c(1:end-1);
c(end) = c(end) + ce;

% extract samples

xv = find(c);

if numel(xv) == n  % each value is sampled at most once
    x = xv;
else                % some values are sampled more than once
    xc = c(xv);
    d = zeros(1, n);
    dv = [xv(1), diff(xv)];
    dp = [1, 1 + cumsum(xc(1:end-1))];
    d(dp) = dv;
    x = cumsum(d);
end

% randomly permute the sample's order
x = x(randperm(n));



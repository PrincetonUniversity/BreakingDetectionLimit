function [Y, placed] = generate_clean_micrograph_2D(X, param)
% Inputs:
%   X: signal of size LxL
%   W: separation zone has size WxW: any square of that size can only touch
%      one signal occurence.
%   N: micrograph has size NxN
%   m: the signal will appear m times in the micrograph, in random places,
%      with separation specified by W. If it proves difficult to place m
%      copies of the signal obeying that rule, the number of copies may be
%      smaller.
%
% Outputs:
%   Y: micrograph of size NxN (clean)
%   placed: actual number of occurrences of the signal X in Y.

% Since we attempt placement at random, we need the end-result to be
% quite sparse to have decent chances of finding m spots.
%if N^2 < 10*m*W^2
%    warning('BigMRA:gendata2D', ...
%        'It may be difficult to get this many repetitions...');
%end

W = param.W;
N = param.N;
m = param.m_want;
rangeW = 0:(W-1);

% The mask has the same size as the micrograph. Each pixel in the mask
% corresponds to a possible location for the upper-left corner of the
% signal, to be placed in the micrograph. A value of zero means the
% spot is free, while a value of 1 means the spot is forbidden (either
% because a signal will already be there, or because it is in the area
% of separation of another signal.)
mask = zeros(N, N);
% The locations table recors the chosen signal locations.
locations = zeros(m, 2);
% This counter records how many signals we successfully placed.
placed = 0;
% Since placement is random, there is a chance that we will get a lot
% of failed candidates. To make up for it, we allow for more than m
% trials. But we still put a hard cap on it to avoid an infinite loop.
%max_trials = 2*m;
max_trials = 4*m;

for counter = 1 : max_trials
    
    % Pick a candidate location for the upper-left corner of the signal
    candidate = randi(N-W+1, 1, 2);
    
    % Check if there is enough room, taking the separation rule into
    % account. That is, a square of size WxW with upper-left corner
    % specified by the candidate must be entirely free.
    if all(mask(candidate(1)+rangeW, candidate(2)+rangeW) == 0)
        
        % Record the successful candidate
        placed = placed + 1;
        locations(placed, :) = candidate;
        
        % Mark the area as reserved
        mask(candidate(1)+rangeW, candidate(2)+rangeW) = 1;
        
        % Stop if we placed sufficiently many signals successfully.
        if placed >= m
            break;
        end
        
    end
    
end

% Now that we have a list of locations, actually go ahead and build the
% micrograph.
L = size(X, 1);
assert(size(X, 2) == L, 'X must be square.');
rangeL = 0 : (L-1);
Y = zeros(N);
%    Y = zeros(N, N,'gpuArray');
for k = 1 : placed
    
    Y(locations(k, 1) + rangeL, locations(k, 2) + rangeL) = X;
    
end

end

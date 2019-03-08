function [y, placed] = generate_clean_micrograph_1D(x, W, N, m)
% Inputs:
%   x: signal of size Lx1
%   W: separation zone has size Wx1: any contiguous subvector of that size
%      can only touch one signal occurrence.
%   N: micrograph has size Nx1
%   m: the signal will appear m times in the micrograph, in random places,
%      with separation specified by W. If it proves difficult to place m
%      copies of the signal obeying that rule, the number of copies may be
%      smaller.
%
% Outputs:
%   y: micrograph of size Nx1 (clean)
%   placed: actual number of occurrences of the signal x in y.

    % Since we attempt placement at random, we need the end-result to be
    % quite sparse to have decent chances of finding m spots.
    if N < 10*m*W
        warning('BigMRA:gendata1D', ...
                'It may be difficult to get this many repetitions...');
    end

    rangeW = 0:(W-1);
    
    % The mask has the same size as the micrograph. Each pixel in the mask
    % corresponds to a possible location for the left-most point of the
    % signal, to be placed in the micrograph. A value of zero means the
    % spot is free, while a value of 1 means the spot is forbidden (either
    % because a signal will already be there, or because it is in the area
    % of separation of another signal.)
    mask = zeros(N, 1);
    % The locations table records the chosen signal locations.
    locations = zeros(m, 1);
    % This counter records how many signals we successfully placed.
    placed = 0;
    % Since placement is random, there is a chance that we will get a lot
    % of failed candidates. To make up for it, we allow for more than m
    % trials. But we still put a hard cap on it to avoid an infinite loop.
    max_trials = 2*m;
    
    for counter = 1 : max_trials
        
        % Pick a candidate location for the left-most point of the signal
        candidate = randi(N-W+1, 1, 1);
        
        % Check if there is enough room, taking the separation rule into
        % account. That is, a sequence of size Wx1 with left-most point
        % specified by the candidate must be entirely free.
        if all(mask(candidate+rangeW) == 0)
            
            % Record the successful candidate
            placed = placed + 1;
            locations(placed) = candidate;
            
            % Mark the area as reserved
            mask(candidate+rangeW) = 1;
        
            % Stop if we placed sufficiently many signals successfully.
            if placed >= m
                break;
            end
            
        end
        
    end
    
    % Now that we have a list of locations, actually go ahead and build the
    % micrograph.
    x = x(:);
    L = size(x, 1);
    rangeL = 0 : (L-1);
    y = zeros(N, 1);
    for k = 1 : placed
        
        y(locations(k) + rangeL) = x;
        
    end

end

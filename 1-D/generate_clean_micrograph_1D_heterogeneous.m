function [y, m_actual] = generate_clean_micrograph_1D_heterogeneous(X, W, n, m)
% Inputs:
%   X: K signals of length L in a matrix of size LxK
%   W: separation zone has size Wx1: any contiguous subvector of that size
%      can only touch one signal occurrence.
%   n: micrograph has size Nx1
%   m: signal k appears m(k) times in the micrograph, in random places,
%      with separation specified by W. If it proves difficult to place m(k)
%      copies of the signal obeying that rule, the number of copies may be
%      smaller. See output m_actual.
%
% Outputs:
%   y: micrograph of size nx1 (clean)
%   m_actual: m_actual(k) is the actual # of occurrences of signal k in y.

    % Since we attempt placement at random, we need the end-result to be
    % quite sparse to have decent chances of finding sum(m) spots.
    if n < 5*sum(m)*W
        warning('BigMRA:gendata1D', ...
                'It may be difficult to get this many repetitions...');
    end

    rangeW = 0:(W-1);
    
    % At first, we pick sum(m) locations without deciding which signal
    % goes where.
    
    % The mask has the same size as the micrograph. Each pixel in the mask
    % corresponds to a possible location for the earliest point of the
    % signal to be placed in the micrograph. A value of zero means the
    % spot is free, while a value of 1 means the spot is forbidden (either
    % because a signal will already be there, or because it is in the area
    % of separation of another signal.)
    mask = zeros(n, 1);
    % The locations table records the chosen signal locations.
    locations = zeros(sum(m), 1);
    % This counter records how many signals we successfully placed.
    placed = 0;
    % Since placement is random, there is a chance that we will get a lot
    % of failed candidates. To make up for it, we allow for more than
    % sum(m) trials. But we still put a hard cap on it to avoid an infinite
    % loop.
    max_trials = 3*sum(m);
    
    for counter = 1 : max_trials
        
        % Pick a candidate location for the earliest point of the signal
        candidate = randi(n-W+1, 1, 1);
        
        % Check if there is enough room, taking the separation rule into
        % account. That is, a sequence of size Wx1 with earliest point
        % specified by the candidate must be entirely free.
        if all(mask(candidate+rangeW) == 0)
            
            % Record the successful candidate
            placed = placed + 1;
            locations(placed) = candidate;
            
            % Mark the area as reserved
            mask(candidate+rangeW) = 1;
        
            % Stop if we placed sufficiently many signals successfully.
            if placed >= sum(m)
                break;
            end
            
        end
        
    end
    
    % Now that we have a list of locations, actually go ahead and build the
    % micrograph. For this, we need to decide which signal goes where.
    % Since it is possible that we did not find sum(m) locations, we first
    % need to review our expectations. One could do a deterministic
    % re-assignment of m, then compute a random permutation. Here, we use a
    % simpler-to-code approach and simply decide randomly what each
    % location contains, proportionally to m.
    [L, K] = size(X);
    rangeL = 0 : (L-1);
    y = zeros(n, 1);
    signal_placement = discretesample(m/sum(m), placed);
    for iter = 1 : placed
        y(locations(iter) + rangeL) = X(:, signal_placement(iter));
    end
    m_actual = hist(signal_placement, 1:K);
    m_actual = reshape(m_actual, size(m));

end

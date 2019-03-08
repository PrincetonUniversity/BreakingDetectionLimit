function P = best_permutation(Xref, X)
% Returns a permutation vector P such that X(:, P) is closest to Xref in
% Frobenius norm. Xref and X must have the same size, and the number of
% columns must be small (the complexity scales with its factorial..)
%
% Nicolas Boumal, May 30, 2018.

    assert(all(size(Xref) == size(X)), 'Xref and X must have same size.');
    
    % Code assumes K is small enough so that this is practical.
    K = size(Xref, 2);
    permutations = perms(1:K);
    
    best_error = inf;
    best_P = [];
    for p = 1 : size(permutations, 1)
        P = permutations(p, :);
        relative_error = norm(Xref-X(:, P), 'fro') / norm(Xref, 'fro');
        if relative_error < best_error
            best_error = relative_error;
            best_P = P;
        end
    end
    P = best_P;
    
end

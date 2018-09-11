function R_init = get_init_random_trials(problem, trials, par_flag)
% Function to try many random guesses for the alignment of two molecules
% and choose the best guess as initialization for Manopt.
% 
% Inputs:
%   * problem: Manopt problem structure for optimization over SO(3)
%   * trials: number of random guesses to check
%   * par_flag: flag set to true to check random guesses in parallel, and
%   false otherwise
% 
% Outputs:
%   * R_init: best of the random guesses
% 
% Eitan Levin, June 2018

if par_flag
    cst = zeros(trials,1); R_init = cell(trials, 1);
    parfor t= 1:trials
        R_init{t} = problem.M.rand();
        cst(t) = problem.cost(R_init{t});
    end
    R_init = R_init{cst == min(cst)};
else
    cst = Inf; R_init = [];
    for t = 1:trials
        R = problem.M.rand();
        cst_R = problem.cost(R);
        if cst_R < cst
            cst = cst_R;
            R_init = R;
        end
    end
end
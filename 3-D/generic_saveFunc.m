function stop = generic_saveFunc(x, optimValues, state, name, iters_per_save)
% Function to save the iterates in the optimization every given number of
% iterations.
% 
% Inputs:
%   * x, optimValues, state: optimization-related variables, see Matlab's
%   documentation. In particular, x is the current iterate.
%   * name: prefix for the filename.
%   * iters_per_save: save an iterate every iters_per_save iterations.
% 
% Outputs:
%   * stop: boolean telling the optimizer whether to terminate or not
% 
% Eitan Levin, August 2018

stop = false;

iter = optimValues.iteration;
if mod(iter, iters_per_save) == 0
    save([name '_iter_' num2str(iter) '_save.mat'], 'x')
end

function C_vec = vec_cell(C)
% Vectorize a cell array
% 
% Inputs:
%   * C: cell array to vectorize
% 
% Outputs:
%   * C_vec: vector obtained by first vecotrizing each cell in C, and then
%   vertical concatenating them.
% 
% Eitan Levin, July 2018

C_vec = cellfun(@(x) x(:), C, 'UniformOutput', 0);
C_vec = vertcat(C_vec{:});

function varargout = ind2sub(siz,ndx)
% Modification of the built-in ind2sub to remove unnecessary sanity checks
% and conversions to double, to accelerate the function - it's called many
% times in the inner loop of some functions. 
% The built-in documentation:
% 
%IND2SUB Multiple subscripts from linear index.
%   IND2SUB is used to determine the equivalent subscript values
%   corresponding to a given single index into an array.
%
%   [I,J] = IND2SUB(SIZ,IND) returns the arrays I and J containing the
%   equivalent row and column subscripts corresponding to the index
%   matrix IND for a matrix of size SIZ.  
%   For matrices, [I,J] = IND2SUB(SIZE(A),FIND(A>5)) returns the same
%   values as [I,J] = FIND(A>5).
%
%   [I1,I2,I3,...,In] = IND2SUB(SIZ,IND) returns N subscript arrays
%   I1,I2,..,In containing the equivalent N-D array subscripts
%   equivalent to IND for an array of size SIZ.
%
%   Class support for input IND:
%      float: double
%   See also SUB2IND, FIND.
 
%   Copyright 1984-2013 The MathWorks, Inc. 

if nargout == 2
    vi = rem(ndx-1, siz(1)) + 1;
    varargout{2} = (ndx - vi)/siz(1) + 1;
    varargout{1} = vi;
else
    k = [1 cumprod(siz(1:end-1))];
    for i = nargout:-1:1,
        vi = rem(ndx-1, k(i)) + 1;
        vj = (ndx - vi)/k(i) + 1;
        varargout{i} = vj;
        ndx = vi;
    end
end

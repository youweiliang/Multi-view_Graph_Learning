function [D]  = extract_from_idx_ONE(idx_matrix, i, j)
% Pick elements from A accroding to idx_matrix with
% A being all one matrix
% Original function is: extract_from_idx(A, idx_matrix)
% Inputs:
%   A - the sourse matrix
%   idx_matrix - an index matrix, pick the elements from A if its colomn index in
%       the corespongding row of idx_matrix
% Outputs:
%   % B - the extracted matrix, same shape as idx_matrix
%   D - the extracted matrix, same shape as A

[m, n] = size(idx_matrix);
rowidx = repmat((1:m)', 1, n);

D = sparse(rowidx, idx_matrix, 1, i, j);

end
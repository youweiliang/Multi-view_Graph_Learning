function [G_std] = std_graph(G, is_knn)
% Standardize a graph with edges of nonnegative weight
% to (almost surely) mean 1 and standard deviation 1
if nargin < 2
    is_knn = false;
end

if ismatrix(G)
    [n,m] = size(G);
    if is_knn
        [I,J,S] = find(G);  % find nonzeros
        sd = std(S);
        mu = mean(S);
        S = (S - mu) / sd + 1;
        S(S < 0) = 0;
        G_std = sparse(I,J,S,n,m);
    else
        S = G(:);
        sd = std(S);
        mu = mean(S);
        S = (S - mu) / sd + 1;
%         idx = S < 0;
%         nz = nnz(idx) / length(S);
%         fprintf('perct: %.2f\n', nz);
        S(S < 0) = 0;
        G_std = reshape(S,n,m);
    end
elseif isvector(G)
    S = G;
    sd = std(S);
    mu = mean(S);
    S = (S - mu) / sd + 1;
%     idx = S < 0;
%     nz = nnz(idx) / length(S);
%     fprintf('perct: %.2f\n', nz);
    S(S < 0) = 0;
    G_std = S;
else
    error('only support matrix or vector standardization')
end

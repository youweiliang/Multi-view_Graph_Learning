function [label, com_a] = SGF(fea, numClust, knn0, w, self_b, cross_b, metric)
% Similarity Graph Learning for multi-view spectral clustering
% Inputs:
%   fea - a cell of feature matrices, each row corresponds to an instance 
%         and each colomn corresponds to a feature
%   numClust - number of desired clusters
%   knn0 - number of k-nearest neighbors
%   w - weight for each view (see our paper)
%   self_b - the hyperparameter beta in our paper
%   cross_b - the hyperparameter gamma in our paper
%   metric - the metric for computing distance matrices, it can be
%            squaredeuclidean, cosine, and original
% Outputs:
%   label - the cluster labels
%   com_a - the affinity matrix of the learned unified graph
% Reference: Multi-view Graph Learning by Joint Modeling of Consistency and Inconsistency


addpath ../MinMaxSelection
addpath ../util
v = length(fea);
tol = 1e-5;
tol2 = 1e-8;
max_iter = 100;

%% decide metric
if nargin < 7
    metric = 'squaredeuclidean';
    if nargin < 6
        cross_b = 1e2;
        if nargin < 5
            self_b = 1;
            if nargin < 4
                w = ones(1, v);
                if nargin < 3
                    knn0 = 20;
                    if nargin < 2
                        error('Not enough inputs.')
                    end
                end
            end
        end
    end
end
knn = knn0 + 1;

%% make distance matrices
WW = make_distance_matrix(fea, metric);

%% preparation
n = size(WW{1}, 1);
knn_idx = logical(sparse(n, n));
idx = cell(1,v);

for i=1:v
    W = WW{i};
    [~, idx{i}] = mink_new(W, knn, 2);
    [tp] = extract_from_idx_ONE(idx{i}, n, n);
    knn_idx = knn_idx | logical(tp);
end
% make knn index symmetric % no need - 2020.7.19
% knn_idx = knn_idx | knn_idx';

[I,J,K] = find(knn_idx);
K(I==J) = false;
knn_idx = sparse(I,J,K,n,n);
ne = nnz(knn_idx);
D = zeros(v, ne);

for i=1:v
    S = WW{i}(knn_idx);
    S = std_graph(S);
    sigma = mean(sqrt(S));  % fixed 2020.7.19
    S = exp(-S/(2*sigma^2));
    D(i,:) = S;
end

b = cross_b*ones(v) - diag(cross_b*ones(1,v)) + diag(self_b*ones(1,v));

%% run graph learning algorithm
[S, call_qp, iter, alpha] = consistent_graph_dca_noMKL_noVS(v, ne, D, b, w, tol, tol2, max_iter);

affinity_matrix = sparse_from_idx(S, knn_idx, n, n);

% do kNN again
com_a = kNN(affinity_matrix, knn0);

[label] = SpectralClustering(com_a, numClust, 3);  % obtain lable for each instance, label # starts from 1


function [label, com_a] = DGF(fea, numClust, knn0, w, self_b, cross_b, metric)

addpath C:\youwei\ConsistentGraphLearning-master\MinMaxSelection
v = length(fea);
tol = 1e-5;
tol2 = 1e-8;
max_iter = 100;

%% decide metric
if nargin < 7
    metric = 'euclidean';  % this actually computes squaredeuclidean
    if nargin < 6
        cross_b = 1e4;
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
if strcmp(metric, 'original')
    WW = fea;
else
    WW = cell(v, 1);
    for i=1:v
        [~, WW{i}] = make_affinity_matrix(fea{i}, metric); % this actually computes squaredeuclidean
    end
end

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

[I,J,K] = find(knn_idx);
K(I==J) = false;
knn_idx = sparse(I,J,K,n,n);
ne = nnz(knn_idx);
D = zeros(v, ne);

for i=1:v
    S = WW{i}(knn_idx);
    S = std_graph(S);
    D(i,:) = S;
    %D(i,:) = D(i,:) / norm(D(i,:));
end

if strcmp(metric, 'euclidean')
    d = 2;
else
    d = 1;
end

b = cross_b*ones(v) - diag(cross_b*ones(1,v)) + diag(self_b*ones(1,v));

[S, call_qp, iter, alpha] = consistent_graph_dca(v, ne, D, b, w, tol, tol2, max_iter);

sigma = mean(S.^(1/d));  % if d==2, this coincides with Gaussian kernel
S = exp(-S/(2*sigma^d));
% % ANOTHER OPTION (but the test results are quite similar in D1-D7)
% sigma = mean(S);
% S = exp(-S/(2*sigma));

affinity_matrix = sparse_from_idx(S, knn_idx, n, n);

% do kNN again
com_a = kNN(affinity_matrix, knn0);

[label] = SpectralClustering(com_a, numClust, 3);  % obtain lable for each instance, label # starts from 1


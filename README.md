# Multi-view Graph Learning
Code for our paper "Multi-view Graph Learning by Joint Modeling of Consistency and Inconsistency", which is currently under review.


### Preparation
* **Windows 64bit**: 
Add some helper files to MATLAB path by `addpath('MinMaxSelection'); addpath('util')` command in MATLAB command window.
* **Linux, Windows 32bit and Mac OS**: 
Add some helper files to MATLAB path by `addpath('MinMaxSelection'); addpath('util')` command in MATLAB command window. Then recompile the helper functions by running `minmax_install` in folder `MinMaxSelection`.


### Example usage
```MATLAB
cd ./consistent_graph_learning
load('../data/UCI_Digits.mat', 'fea', 'gt')  % load features and ground truth
numClust = length(unique(gt)); 
[label, com_a] = SGF(fea, numClust);  % clustering labels and the learned consistent graph
score = getFourMetrics(label, gt)  % 4 metrics: ACC, ARI, NMI, purity
[label, com_a] = DGF(fea, numClust);
score = getFourMetrics(label, gt)
```

### Multi-view Data
More multi-view data are available on [this Google Drive](https://drive.google.com/drive/folders/1vzJ19eGy7sAyLTFtM4IWkKzZhFJsi134?usp=sharing "multi-view data").

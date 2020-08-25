# Multi-view Graph Learning
Code for our paper "[Multi-view Graph Learning by Joint Modeling of Consistency and Inconsistency](https://arxiv.org/abs/2008.10208)", which is follow-up work of our ICDM paper "Consistency Meets Inconsistency: A Unified Graph Learning Framework for Multi-view Clustering".


### Preparation
* Using with MATLAB
  * **Windows 64bit**  
  Add some helper files to MATLAB path by `addpath('MinMaxSelection'); addpath('util')` command in MATLAB command window. In subfolder `consistent_graph_learning`, rename one of the files `consistent_graph_dca_MKL_VS` `consistent_graph_dca_MKL` to `consistent_graph_dca` if you have MKL and/or Visual Studio installed. The default file `consistent_graph_dca` assumes **no MKL and no Visual Studio** installed. It is highly recommended to install MKL for better performance (it's free).
  * **Linux, Windows 32bit and Mac OS**  
  Step 1. Add some helper files to MATLAB path by `addpath('MinMaxSelection'); addpath('util')` command in MATLAB command window.
  Step 2. Recompile the helper functions by running `minmax_install` in folder `MinMaxSelection`.
  Step 3. Recompile `consistent_graph_dca` following guidance in [this README](https://github.com/youweiliang/Multi-view_Graph_Learning/tree/master/graph_learning_cpp).
* Using with C++ or C  
For folks without MATLAB and for industrial usage, the Consistent Graph Learning algorithm is coded in C++ (or C with slight modification). An MATLAB implementation should be simple following the algorithm in the paper and may be faster than a C++ implementation without MKL, since the bottleneck of the Consistent Graph Learning algorithm lies in large matrix multiplication and MATLAB internally uses MKL for matrix multiplication. Therefore, if using C++ implementation, it is highly recommended to install MKL (it's free). See [this README](https://github.com/youweiliang/Multi-view_Graph_Learning/tree/master/graph_learning_cpp).

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

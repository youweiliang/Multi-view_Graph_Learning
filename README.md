# Multi-view Graph Learning
Code for our paper "[Multi-view Graph Learning by Joint Modeling of Consistency and Inconsistency](https://arxiv.org/abs/2008.10208)", which is the follow-up work of our ICDM paper "Consistency Meets Inconsistency: A Unified Graph Learning Framework for Multi-view Clustering". ***This version is more robust and stable and efficient than the ICDM version.*** Please see the paper for details. 

### News
* [Sep, 2020]: The code of the 7 multi-view spectral clustering algorithms (and a single-view spectral clustering algorithm) used for comparison in our paper is uploaded to [this repository](https://github.com/youweiliang/Multi-view_Clustering). 

### Preparation
* Using with MATLAB
  * **Windows 64bit**
    * Add some helper files to MATLAB path by `addpath('MinMaxSelection'); addpath('util')` command in MATLAB command window.
    * In folder `consistent_graph_learning`, rename one of the files `consistent_graph_dca_MKL_VS.mexw64` `consistent_graph_dca_MKL.mexw64` to `consistent_graph_dca.mexw64` if you have MKL and/or Visual Studio installed. The default file `consistent_graph_dca.mexw64` assumes _no MKL and no Visual Studio_ installed. If your dataset size is larger than 20,000, it is highly recommended to install MKL for better performance (it's free).
  * **Windows 32bit, Linux, Mac OS**  
    - Add some helper files to MATLAB path by `addpath('MinMaxSelection'); addpath('util')` command in MATLAB command window.
    - Recompile the helper functions by running `minmax_install` in folder `MinMaxSelection`.
    - Recompile `consistent_graph_dca` following guidance in [this README](https://github.com/youweiliang/Multi-view_Graph_Learning/tree/master/graph_learning_cpp). And move compiled file `consistent_graph_dca.mex???` (the filename extension depends on your system) to folder `consistent_graph_learning`.
* Using with C++ or C  
For folks without MATLAB and for industrial usage, the Consistent Graph Learning algorithm is coded in C++ (or C with slight modification). So, it should be trivial to integrate it into a C++ or C application. **Also, a pure MATLAB implementation should be simple by following the algorithm in the paper and would be as faster as a C++ implementation _with_ MKL, since the computational bottleneck of the Consistent Graph Learning lies in large matrix multiplication and MATLAB internally uses MKL for matrix multiplication. Therefore, if using C++ or C implementation and the dataset size is larger than 20,000, it is highly recommended to install MKL (it's free).** See [this README](https://github.com/youweiliang/Multi-view_Graph_Learning/tree/master/graph_learning_cpp).

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

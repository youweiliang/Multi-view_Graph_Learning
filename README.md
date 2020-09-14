# Multi-view Graph Learning
Code for our paper "[Multi-view Graph Learning by Joint Modeling of Consistency and Inconsistency](https://arxiv.org/abs/2008.10208)", which is the follow-up work of our ICDM paper "Consistency Meets Inconsistency: A Unified Graph Learning Framework for Multi-view Clustering". ***This version is more robust and stable and efficient than the ICDM version.*** Please see the paper for details. 

### News
* [Sep, 2020]: The code of the 7 multi-view spectral clustering algorithms (and a single-view spectral clustering algorithm) used for comparison in our paper is uploaded to [this repository](https://github.com/youweiliang/Multi-view_Clustering). 
* [Sep, 2020]: All datasets used in our paper are uploaded to Baidu Cloud and Google Drive. 

### Dataset
All datasets used in our paper are available at [Baidu Cloud](https://pan.baidu.com/s/1bAfDcgH3NguqWM6saDTv1g) with code `pqti` and [Google Drive](https://drive.google.com/drive/folders/1UtjL0Og7ALs9AJq9XnkdrYUmr5rudCyk?usp=sharing). Each dataset is a mat file containing 2 variables `fea` (i.e., a MATLAB cell of features) and `gt` (i.e., ground truth label), except the file `flower17.mat` which contains a cell of distance matrices and ground truth since features are unavailable. 

### Preparation
* Using with MATLAB
  * **Windows 64bit**
    * Add some helper files to MATLAB path by `addpath('MinMaxSelection'); addpath('util')` command in MATLAB command window.
    * In the folder `consistent_graph_learning`, according to whether there is MKL or Visual Studio (VS) installed on your machine, choose one among the three files `consistent_graph_dca_noMKL_noVS.mexw64`, `consistent_graph_dca_MKL_VS.mexw64`, and `consistent_graph_dca_MKL_noVS.mexw64`. And rename the chosen file to `consistent_graph_dca.mexw64`. The default file `consistent_graph_dca.mexw64` is just `consistent_graph_dca_noMKL_noVS.mexw64`, which assumes _no MKL and no Visual Studio_ installed. If your dataset size is larger than 20,000, it is highly recommended to install MKL for better performance (it's free).
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

### Citation
```
@article{liang2020multi,
  title={Multi-view Graph Learning by Joint Modeling of Consistency and Inconsistency},
  author={Liang, Youwei and Huang, Dong and Wang, Chang-Dong and Yu, Philip S},
  journal={arXiv preprint arXiv:2008.10208},
  year={2020}
}
```

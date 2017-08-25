# LeToR_Linear_Regression
Implementation of  linear regression using closed form solution and SGD to solve Learning to Rank (LeToR) problem in Information Retrieval

## Introduction
The goal of this project is to use machine learning to solve a problem that arises in Information Retrieval, one known as the Learning to Rank (LeToR) problem. 

We have 4 subtasks to solve, which are as follows: 

    1. Train a linear regression model on LeToR dataset using a closed-form solution. 
    
    2. Train a linear regression model on the LeToR dataset using stochastic gradient descent (SGD). 
    
    3. Train a linear regression model on a synthetic dataset using a closed-form solution. 
    
    4. Train a linear regression model on the synthetic dataset using SGD.

## Dataset

  1. **[LeToR (Learning to Rank) Dataset](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)**

  2. Synthetic generated dataset
 
## Implementation

First, we split the dataset into Train-Validation-Test sets.

We apply **linear regression with Gaussian basis function** on this data set, using two approaches: 
   1. Closed Form Solution 
   2. Stochastic Gradient Descent Solution

Real world data is not linear, and has elements of non linearity in it, which cannot be modeled using linear functions. 

We fit **RBF kernels** on the training data, to introduce and account for the non-linearity component in the data. From there, once we have multiple Gaussian basis functions, the task is to fit a linear model for each of these functions.

Choosing Hyperparameters:
  1. Number of RBFs to fit (M)
  
  2. Centers of RBF Kernels (μj) using K-Means Clustering
  
  3. Radius/Spread of Gaussian Kernels (Σ)
  
## Results

**LeToR Dataset**
  1. Using **Closed form Solution**, we achieve an **RMSE of 0.569938986664** on Test Set. Best model has 30 RBF Kernels and λ = 0.01

  2. Using **SGD Solution**, we achieve an **RMSE of 0.637591911743** on Test Set. Best model has 11 RBF Kernels and λ = 0.01
  
**Synthetic Dataset**
  1. Using **Closed form Solution**, we achieve an **RMSE of 0.703795980591** on Test Set. Best model has 8 RBF Kernels and λ = 0.1

  2. Using **SGD Solution**, we achieve an **RMSE of 0.789271504073** on Test Set. Best model has 5 RBF Kernels and λ = 0.1


## Conclusions

1. On comparing the performance of the two methods we observe that for the **closed form solution** approach, we need to calculate NxM design matrix for each values of M. So that becomes approximately N x M x M. However for small number of features, M is a constant and run time will be in the **O(N)** .

2. In comparison, for **stochastic gradient descent** approach we need to calculate n x M order design matrix. (n<N, where n is the value when the error change becomes minimal). However if we have a design matrix already computed the runtime is of O(n x M). Computing the whole design matrix, or saving the already computed values iteratively, results in a much faster implementation of the gradient descent approach and will be much **faster than closed form solution approach as n<N**, for n being the value when we stop the weight matrix iterations when the error change become significantly small.


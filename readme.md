# Secure Distributed DPHelmet

This archive contains code to reproduce the `DP_SGD_SVM` accuracy numbers for the paper "Single SMPC Invocation DPHelmet: Differentially Private Distributed Learning on a Large Scale".

As we rely on output perturbation, we add multivariate Gaussian noise to the SVMs after running SGD_SVM. Hence, we provide the code that we used for training the SGD_SVM (see `SingleInvocationDPHelmet.py`). As discussed in `Multivariate noise calibration`, we ran the PrivacyBuckets to get tight ($\varepsilon$, $\delta$)-DP bounds for the multivariate Gaussian Mechanism. For non-subsampled Gaussians, we ran `dphelmet_tight_adp.py` instead which is considerably faster.

## Requirements
* We used Python 3.x.
* We used Tensorflow 2.3.1.

## Instructions

Run `extract_embeddings.py` first which creates two embedding files: `code_space.npy` as well as `labels.npy`.
Afterward run `SingleInvocationDPHelmet.py` to train the distributed `DP_SGD_SVM` which generates a file `tests_dphelmet_<datetime>.csv` listing all experiment results i.e., the accuracies as well as f1-scores (macro) after noise. For generating figures 3,4, and 5 of the paper for the DPHelmet variant only, refer to `example_viz.py`. Note that this part requires adaption dependent on the particular experiment configuration in use.

### Resources / Runtimes
Expect this algorithm to take some time. In addition to the CIFAR-10 dataset (~162MB), it has to download about 3.3GB for the pre-trained model.
Extracting the embeddings is highly dependent on your GPU resources and takes about an hour with good resources. Try lowering the batch size if too much GPU-RAM is consumed.

Running the cross-validation search of distributed DP-Helmet requires some additional CPU resources. Expect about 10min for one parameter configuration (for 1000 users and 50 data points each).
The current default is 48 hyperparameter configurations with 12 runs each. Thus it is highly recommended to parallelize by changing the `N_PROCESSES` to your liking.

The last part of the cross-validation search (where the noise is added) additionally consumes a few hours depending on the number of parameters (this part is not parallelized).

## Reconstructing DP-FL

For DP-FL, we are gracious and assume a noise overhead of only sqrt(#users), as we are not aware of any techniques (short of SMPC) that achieve less than sqrt(#users) noise overhead.

We did not automate the DP-FL code. We only use standard tools, though. Here are the steps that are needed to reconstruct the DP-FL results.

1. Extract the CIFAR-10 embeddings via `extract_embeddings.py`.
2. Download and install opacus v0.15.0 (e.g. via `pip3 install opacus==0.15.0`)
3. Run our DP-FL code (modify `sigma`-hyperparameter for different privacy budgets). Results are saved at `run_results_***.npy`.

        python3 dpsgd_cifar10_opacus.py

4. Find a better privacy budget by running the provided privacy bucket program.

        python3 privacy_buckets_dpsgd.py


## Multivariate Gaussian noise calibration

In the code where we construct the `DP_SGD_SVM` classification accuracy results, we use fixed ($\varepsilon$, $\delta$) pairs for the Gaussian Mechanism. We computed those with the PrivacyBuckets tool. Below, we describe how to reconstruct those numbers.

The PrivacyBuckets tool provides tight sequential composition bounds for one-dimensional output perturbation mechanisms. As a d-dimensional spherical multivariate Gaussian distribution (i.e., with a diagonal covariance matrix) can be represented as product distribution (i.e., the joint distribution) of d identical (and independent) 1-dimensional Gaussian distributions, the leakage of the spherical multivariate Gaussian mechanism, is the same as the sequential composition of 1-dimensional Gaussian distributions.

To reproduce our results, please follow these steps:

1) Run our PrivacyBuckets script

        python3 privacy_buckets_dphelmet.py

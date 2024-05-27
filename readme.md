# Distributed DPHelmet

This archive contains code to reproduce `DP-SVM-SGD` and `DP-Softmax-SLP-SGD` of the [Distributed DP-Helmet: Scalable Differentially Private Non-interactive Averaging of Single Layers](https://arxiv.org/abs/2211.02003) paper.

The main code is in
- distributed `DP-SVM-SGD`: [DPHelmet.py](DPHelmet.py) and
- distributed `DP-Softmax-SLP-SGD`: [DPHelmet_softmax.py](DPHelmet_softmax.py).

We also provide code for
- the SimCLR-based feature extraction: [extract_embeddings.py](extract_embeddings.py), [extract_embeddings_emnist.py](extract_embeddings_emnist.py) and [simclr_preprocessing.py](simclr_preprocessing.py),
- tight ($\varepsilon$, $\delta$)-DP accounting for our algorithm: [DPHelmet_tight_adp.py](DPHelmet_tight_adp.py), and
- code for related work `DP-FL` incl. tight accounting: [dpsgd_cifar10_opacus.py](dpsgd_cifar10_opacus.py) and [privacy_buckets_dpsgd.py](privacy_buckets_dpsgd.py).

## Instructions

### Requirements
* Python 3.x.
* Tensorflow 2.x.

### Supported Datasets
- CIFAR-10: run [extract_embeddings.py](extract_embeddings.py) which outputs `code_space_cifar10.npy` and `labels_cifar10.npy`
- CIFAR-100: run [extract_embeddings.py](extract_embeddings.py) with `DATASET_NAME = cifar100` which outputs `code_space_cifar100.npy` and `labels_cifar100.npy`
- federated EMNIST: run [extract_embeddings_emnist.py](extract_embeddings_emnist.py) which outputs `code_space_federated_emnist.npy`, `labels_federated_emnist.npy`, and `userid_federated_emnist.npy`

Use the `DATASET` variable to switch between these datasets while running distributed `DP-SVM-SGD` or `DP-Softmax-SLP-SGD`.

### Output

Both `DP-SVM-SGD` and `DP-Softmax-SLP-SGD`, generate a file `tests_dphelmet_<datetime>.csv` listing all experiment results, i.e. the hyperparameters, the accuracies, and the f1-scores (macro) after noise. For generating some experimental figures of the paper for the DPHelmet variant only, refer to [example_viz.py](example_viz.py). Note that the visualization part requires adaption dependent on the used experiment configuration.

Note that the $(\varepsilon,\delta)$ output in `tests_dphelmet_<datetime>.csv` are only estimates. For exact bounds, run [DPHelmet_tight_adp.py](DPHelmet_tight_adp.py). Make sure to specify the `EPS_TARGET` constant with the value of interest in the `dp_eps` column in the CSV file. You can also modify current $\delta = 10^{-5}$ with the `DELTA_TARGET` constant.

### Resources / Runtimes
Expect this algorithm to take some time. In addition to the CIFAR-10 dataset (~162MB), CIFAR-100 dataset, or federated EMNIST dataset, it has to download about 3.3GB for the pre-trained model.
The speed of extracting the embeddings depends on your GPU resources and takes about an hour with good resources. Try lowering the batch size if too much GPU-RAM is consumed.

Running the cross-validation search of distributed DP-Helmet requires some additional CPU-only resources which depend on the number of hyperparameters, the number of epochs, and the number of runs. It is highly recommended to parallelize by changing the `N_PROCESSES` to your liking.

The last part of the cross-validation search (where the noise is added) additionally consumes a few hours depending on the number of parameters. This part is not parallelized.

## Reconstructing Related Work (DP-FL)

For DP-FL, we are gracious and assume a noise overhead of only $\sqrt{w}$ for $w$ users, as we are not aware of any techniques (short of MPC) that achieve less than $\sqrt{w}$ noise overhead.

We did not automate the DP-FL code. We only use standard tools, though. Here are the steps that are needed to reconstruct the DP-FL results.

1. Extract the embeddings as detailed in [Supported Datasets](#supported-datasets)
2. Install opacus v0.15.0 (e.g. via `pip3 install opacus==0.15.0`)
3. Run our DP-FL code (modify the `sigma`-hyperparameter for different privacy budgets). Results are saved at `run_results_***.npy`.

        python3 dpsgd_cifar10_opacus.py

4. Find a better privacy budget by running the provided privacy bucket program.

        cd privacybuckets
        git submodule update --init --recursive
        cd -
        python3 privacy_buckets_dpsgd.py


## Cite
```
@misc{kirschte2024distributed,
      title = Distributed DP-Helmet: Scalable Differentially Private Non-interactive Averaging of Single Layers,
      author = Moritz Kirschte and Sebastian Meiser and Saman Ardalan and Esfandiar Mohammadi,
      year = 2024,
      eprint = 2211.02003,
      archivePrefix = arXiv,
      primaryClass = cs.CR
}
```
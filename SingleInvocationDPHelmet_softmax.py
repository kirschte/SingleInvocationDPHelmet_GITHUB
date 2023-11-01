#!/usr/bin/env python
# coding: utf-8

import functools
import itertools
import os
import time
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

tf.compat.v1.enable_v2_behavior()
print(tf.__version__)

# NB, This code does not work with multi-process GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # DO NOT MODIFY.

DATASET = 'CIFAR10'

SAVED_EMBEDDINGS_PTH = "./"
SAVED_EMBEDDINGS_FILENAME = "code_space.npy" if DATASET == 'CIFAR10' else "code_space_100.npy"
SAVED_LABELS_FILENAME = "labels.npy" if DATASET == 'CIFAR10' else "labels_100.npy"


####################
### 1. SimCLR Embedding Extraction (execute `extract_embeddings.py` first)
####################

N_CLASSES = 10 if DATASET == 'CIFAR10' else 100

# you need to execute `extract_embeddings.py` first
code_space = np.load(os.path.join(SAVED_EMBEDDINGS_PTH, SAVED_EMBEDDINGS_FILENAME))
labels = np.load(os.path.join(SAVED_EMBEDDINGS_PTH, SAVED_LABELS_FILENAME))

# clip inputs
X_norm = np.linalg.norm(code_space, ord=2, axis=1)
if DATASET == 'CIFAR10':
    clip_bound = 34.854 - 1e-5  # 95.5-percentile of CIFAR-100 embeddings
else:
    clip_bound = 34.157 - 1e-5  # 95.5-percentile of CIFAR-10 embeddings
X_clip = (
    code_space / np.where(X_norm > clip_bound, X_norm / clip_bound, 1)[:, np.newaxis]
)
clip_bound += 1e-5
print(f"{np.linalg.norm(X_clip, ord=2, axis=1).max():.6f} <~= {clip_bound}")


####################
### 2. Distributed DP-Helmet
####################


def evaluate_distributed_psgd(
    X_train,
    y_train,
    n_classes,
    clip_bound,
    lambda_=100,
    bs=20,
    l2=0.07,
    epochs=90,
    n_users=100,
    n_per_user=500,
):
    """Train DP_SGD_SVM. This is the version used in the paper (Algorithm 2).

    Args:
        X_train (np.array): input dataset (features).
        y_train (np.array): input dataset (labels).
        n_classes (int): number of classes.
        clip_bound (float): norm clipping bound of X_train.
        lambda_ (float, optional): regularization parameter of the SVM. Defaults to 100.
        bs (int, optional): batch size of SGD update. Defaults to 20.
        l2 (float, optional): model clipping bound: "l2-projection" (called R in the paper). Defaults to 0.07.
        epochs (int, optional): number of training epochs. Defaults to 90.
        n_users (int, optional): number of users. Defaults to 100.
        n_per_user (int, optional): number of data points per user. Defaults to 500.

    Returns:
        (list, list, float): Triple of (1) the SVM coefficients with shape (n_users, (n_classes, n_features)),
                             (2) the SVM intercept (i.e. bias) with shape (n_users, (n_classes)) and
                             (3) the maximal actual radius (i.e. l2 norm of the SVM parameters)
                             which is NON-PRIVATE but useful for debug purposes.
    """
    d = X_train.shape[1]  # dimensions
    beta = lambda_ + clip_bound**2  # beta smoothness
    beta = np.sqrt( 0.5*beta**2 + n_classes * (d+1) * lambda_**2 )  # correct for higher dimensions

    # prepare inputs
    y_train_onehot = tf.constant(np.eye(n_classes)[y_train].T, dtype=tf.float32)
    inputs = tf.constant(X_train, dtype=tf.float32)
    h = tf.constant(h, dtype=tf.float32)
    lambda_ = tf.constant(lambda_, dtype=tf.float32)

    @tf.function
    def J(c, i, x, y, l):
        """The SVM training objective.

        Args:
            c (np.array): SVM coefficients.
            i (np.array): SVM intercept.
            x (np.array): input dataset (features).
            y (np.array): input dataset (one-hot-encoded labels).
            l (float): regularization parameter $\lambda$.

        Returns:
            np.array: the loss.
        """
        z = (tf.matmul(c, x, transpose_b=True) + i[:, None])
        return 0.5 * l * tf.reduce_sum(
            tf.linalg.diag_part(tf.matmul(c, c, transpose_b=True)) + i**2
        ) + tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(y), logits=tf.transpose(z))
        )

    n_iter_per_epoch = n_per_user // bs + (0 if n_per_user % bs == 0 else 1)
    coefs, intercepts, radius = [], [], []
    for n in range(n_users):

        # initialize hyperplane + intercept
        coef = tf.Variable(
            tf.keras.initializers.Zeros()((n_classes, d)),
            dtype=tf.float32,
            trainable=True,
        )  # zeros init
        intercept = tf.Variable(
            tf.keras.initializers.Zeros()((n_classes,)),
            dtype=tf.float32,
            trainable=True,
        )  # zeros init

        # assign data to users
        inputs_, y_train_onehot_ = (
            inputs[n * n_per_user : (n + 1) * n_per_user],
            y_train_onehot[:, n * n_per_user : (n + 1) * n_per_user],
        )

        for i in range(epochs):
            # shuffle data
            new_idx = tf.random.shuffle(tf.range(n_per_user))
            inputs_, y_train_onehot_ = tf.gather(inputs_, new_idx), tf.gather(
                y_train_onehot_, new_idx, axis=1
            )
            for j in range(n_iter_per_epoch):
                # select batch data
                batch_idx = tf.range(n_per_user)[j * bs : (j + 1) * bs]
                inputs__, y_train_onehot__ = tf.gather(inputs_, batch_idx), tf.gather(
                    y_train_onehot_, batch_idx, axis=1
                )

                # calculate loss
                with tf.GradientTape() as tape:
                    tape.watch([coef, intercept])
                    loss = tf.reduce_mean(
                        J(coef, intercept, inputs__, y_train_onehot__, l=lambda_)
                    )

                # SGD update step
                delta_J_c, delta_J_i = tape.gradient(loss, [coef, intercept])
                lr_ = tf.minimum(
                    1 / lambda_ * 1 / (i * n_iter_per_epoch + j + 1), 1 / beta
                )
                coef = coef - lr_ * delta_J_c
                intercept = intercept - lr_ * delta_J_i

                # make l2-projection with radius `l2`
                actual_l2 = tf.maximum(
                    l2, tf.sqrt(tf.norm(coef) ** 2 + tf.norm(intercept) ** 2)
                )
                coef = coef / (actual_l2 / l2)
                intercept = intercept / (actual_l2 / l2)
                
        coefs.append(coef.numpy())
        intercepts.append(intercept.numpy())
        radius.append(
            tf.reduce_max(tf.sqrt(tf.norm(coef) ** 2 + tf.norm(intercept) ** 2))
        )  # (optionally) track non-DP radius

    return coefs, intercepts, np.max(radius)


####################
### 3. Cross-Validation
####################
# First train the hyperplanes, then noise them depending on `eps`.


### CV-PARAMS ###
NB_SPLITS = 6
NB_REPEATS = 5
N_RUNS = NB_SPLITS * NB_REPEATS
N_PROCESSES = 2
### CV-PARAMS (END) ###

tests_dphelmet_pre = pd.DataFrame(
    columns=[
        "variant",
        "coefs",
        "intercepts",
        "test_indices",
        "unnoised_radius",
        "lambda",
        "bs",
        "l2",
        "epochs",
        "n_users",
        "n_per_user",
    ]
)


def multi_eval(configuration, n_classes, clip_bound, X_clip, labels, noniid):
    """wrapper for multi-process evaluation

    Args:
        configuration (((np.array, np.array), list)): selected training configuration incl. training as well as
                                                      testing indicies and also model parameters.
                                                      Model parameters are: (regularization lambda, batch_size,
                                                       smoothness h, radius R, n_epochs, n_users, n_per_user).
        n_classes (int): number of classes.
        clip_bound (float): norm clipping bound of X_clip.
        X_clip (np.array): clipped input dataset (features).
        labels (np.array): input dataset (labels).
        noniid (bool): setup data among users in a strongly-biased non-iid setting.

    Returns:
        dict: A dictionary containing the training configuration as well as the trained SVM.
    """
    (train_index, test_index), params = configuration

    X_train, y_train = X_clip[train_index], labels[train_index]
    if noniid:
        idx = np.argsort(y_train)
        X_train, y_train = X_train[idx], y_train[idx]

    coefs, intercepts, radius = evaluate_distributed_psgd(
        X_train,
        y_train,
        n_classes=n_classes,
        clip_bound=clip_bound,
        lambda_=params[0],
        bs=int(params[1]),
        l2=params[2],
        epochs=int(params[3]),
        n_users=int(params[4]),
        n_per_user=int(params[5]),
    )
    return {
        "coefs": coefs,
        "intercepts": intercepts,
        "radius": radius,  # NON-PRIVATE, debug purposes only.
        "test_indices": test_index,
        "lambda": params[0],
        "bs": int(params[1]),
        "l2": params[2],
        "epochs": int(params[3]),
        "n_users": int(params[4]),
        "n_per_user": int(params[5]),
    }


### HYPERPARAMS ###
LAMBDA = [1, 3, 10, 30]  # regularization parameter
BS = [20]  # batch size
L2 = [0.1, 0.4, 0.6, 1.0]  # radius R, non-dep. on LAMBDA
EPOCHS = [100]  # epochs
N_USERS = [1, 10, 100]  # number of users
N_PER_USER = [500]  # number of data points per user
NONIID = False
### HYPERPARAMS (END) ###


# prepare hyperparams search space
param_test = np.array(
    list(itertools.product(LAMBDA, BS, L2, EPOCHS, N_USERS, N_PER_USER))
)
# > make sure that not more datapoints are used than there are accessible
param_test = param_test[
    param_test[:, 4] * param_test[:, 5] <= len(code_space) * (NB_SPLITS - 1) / NB_SPLITS
]
print(f">> testing {len(param_test)} parameter combination(s)")

# cross-validation technique
validator = RepeatedStratifiedKFold(
    n_splits=NB_SPLITS, n_repeats=NB_REPEATS * len(param_test)
)

# pre-instanciate training routine
my_multi_eval = functools.partial(
    multi_eval,
    n_classes=N_CLASSES,
    clip_bound=clip_bound,
    X_clip=X_clip,
    labels=labels,
    noniid=NONIID,
)

with Parallel(n_jobs=N_PROCESSES, verbose=40) as p:
    # run DP_Softmax_SLP_SGD in parallel for the hyperparams search space
    scores = p(delayed(my_multi_eval)(conf)
        for conf in zip(
            validator.split(X_clip, labels),
            param_test[None].repeat(N_RUNS, axis=0).reshape(-1, 6),
        ))
    # store the experiment results
    tests_dphelmet_pre = pd.concat([tests_dphelmet_pre, pd.DataFrame(
        [
            {
                "variant": "dist_dphelmet",
                "bs": score["bs"],
                "lambda": score["lambda"],
                "l2": score["l2"],
                "epochs": score["epochs"],
                "coefs": score["coefs"],
                "intercepts": score["intercepts"],
                "unnoised_radius": score["radius"],
                "test_indices": score["test_indices"],
                "n_users": score["n_users"],
                "n_per_user": score["n_per_user"],
            }
            for score in scores
        ], columns=tests_dphelmet_pre.columns)],
        ignore_index=True,
    )

tests_dphelmet = pd.DataFrame(
    columns=[
        "variant",
        "test_acc",
        "test_f1",
        "unnoised_radius",
        "dp_eps",
        "dp_delta",
        "lambda",
        "bs",
        "h",
        "l2",
        "epochs",
        "n_users",
        "n_per_user",
    ]
)

### PRIVACY PARAMETERS ###
EPS = [0.1, 0.2, 0.5, 0.8, 1, 1.5, 2, 5, 10, 100]  # these are only eps estimates
DELTA = 1e-5  # changing this requires a re-run of privacy buckets
### PRIVACY PARAMETERS (END) ###

for eps in EPS:
    # for each hyperplane add noise and predict, dependent on eps and delta
    test_accs, test_f1s = [], []
    for i, x in tests_dphelmet_pre.iterrows():
        # > This is only a noise scale estimate.
        # > For a correct eps refer to the `Gaussian mechanism` (Lemma 6) or Privacy Buckets
        noise_scale = (
            (
                2 / x["lambda"] * (np.sqrt(2)*clip_bound + x["l2"] * x["lambda"]) / (x["n_per_user"])
            )  # sensitivity
            # for Corollary 14 use `2 * x['l2']` as a sensitivity instead
            * np.sqrt(2 * np.log(1.25 / DELTA))  # estimate c for Gaussian leakage
            / (eps * np.sqrt(x["n_users"]))  # cf. Corollary 7 with eps := sigma.
        )

        # 50%-non-colluding assumption
        if x["n_users"] > 1:  # does not make sense for 1 user...
            noise_scale *= np.sqrt(2)  # t=0.5

        coefs, intercepts = [], []
        for u in range(int(x["n_users"])):
            # noise the hyperplane plus intercept
            this_coef, this_intercept = x["coefs"][u], x["intercepts"][u]
            coef_noised = this_coef + np.random.normal(
                loc=0, scale=noise_scale, size=this_coef.shape
            )
            intercept_noised = this_intercept + np.random.normal(
                loc=0, scale=noise_scale, size=this_intercept.shape
            )

            # make l2-projection with radius `l2`
            actual_l2 = tf.maximum(
                x["l2"],
                tf.sqrt(tf.norm(coef_noised) ** 2 + tf.norm(intercept_noised) ** 2),
            )
            coef_noised = coef_noised / (actual_l2 / x["l2"])
            intercept_noised = intercept_noised / (actual_l2 / x["l2"])

            coefs.append(coef_noised)
            intercepts.append(intercept_noised)

        # take the averaged hyperplanes across users + predict
        coef_, intercept_ = tf.reduce_mean(tf.stack(coefs), axis=0), tf.reduce_mean(
            intercepts, axis=0
        )
        y_pred = tf.argmax(
            tf.matmul(coef_, X_clip[x["test_indices"]], transpose_b=True)
            + intercept_[:, None],
            axis=0,
        ).numpy()
        test_acc = accuracy_score(labels[x["test_indices"]], y_pred)
        test_f1 = f1_score(labels[x["test_indices"]], y_pred, average="macro")
        test_accs.append(test_acc)
        test_f1s.append(test_f1)

    # store the experiment results incl. test accuracy and f1-score (macro)
    tests_dphelmet = pd.concat([tests_dphelmet, pd.DataFrame(
        [
            {
                "variant": x["variant"],
                "bs": x["bs"],
                "lambda": x["lambda"],
                "dp_eps": eps,
                "dp_delta": DELTA,
                "h": -1,
                "l2": x["l2"],
                "epochs": x["epochs"],
                "test_acc": test_accs[i],
                "test_f1": test_f1s[i],
                "unnoised_radius": x["unnoised_radius"],
                "n_users": x["n_users"],
                "n_per_user": x["n_per_user"],
            }
            for i, x in tests_dphelmet_pre.iterrows()
        ], columns=tests_dphelmet.columns)],
        ignore_index=True,
    )

# save prediction to .csv file
filename = f"tests_dphelmet_{time.strftime('%Y%m%d_%H%M%S')}.csv"
tests_dphelmet.to_csv(filename, index=False)
print("Written output to", filename, "with scenario noniid", NONIID, "and dataset", DATASET)

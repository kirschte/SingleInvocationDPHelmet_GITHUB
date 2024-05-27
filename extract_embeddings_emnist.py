#!/usr/bin/env python
# coding: utf-8

import os
import collections

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_federated as tff

from simclr_preprocessing import preprocess_image

tf.compat.v1.enable_v2_behavior()
print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # You can activate GPU here

SAVED_EMBEDDINGS_PTH = "./"
SAVED_EMBEDDINGS_FILENAME = "code_space_federated_emnist.npy"
SAVED_LABELS_FILENAME = "labels_federated_emnist.npy"
SAVED_USERID_FILENAME = "userid_federated_emnist.npy"


####################
### 1. SimCLR Embedding Extraction
####################

BATCH_SIZE = 8  # for SimCLR prediction only

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=False)


def _preprocess(image):
    img = tf.repeat(tf.reshape(image["pixels"], [28, 28, 1]), 3, axis=-1)
    return collections.OrderedDict(
        x=preprocess_image(1.0 - img, 224, 224, is_training=False, test_crop=True),
        y=image["label"],
    )


def prepare_dataset(client_data, client_ids):
    return [
        client_data.create_tf_dataset_for_client(x).map(_preprocess).batch(BATCH_SIZE)
        for x in client_ids
    ]


ds_train = prepare_dataset(emnist_train, emnist_train.client_ids)
ds_test = prepare_dataset(emnist_test, emnist_test.client_ids)

SAVED_MODEL_PTH = (
    "gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r152_3x_sk1/saved_model/"
)
saved_model = tf.saved_model.load(SAVED_MODEL_PTH)


code_space_train, code_space_test = [], []
labels_train, labels_test = [], []
userid_train, userid_test = [], []

# train len: 671585
for i, id in enumerate(emnist_train.client_ids):
    for x in ds_train[i]:
        code_space_train.append(saved_model(x["x"], trainable=False)["final_avg_pool"])
        labels_train.append(x["y"])
        l = len(x["y"])
        userid_train.append(l * [id])
code_space_train = np.concatenate(code_space_train)
labels_train = np.concatenate(labels_train)
userid_train = np.concatenate(userid_train)

# test len: 77483
for i, id in enumerate(emnist_test.client_ids):
    for x in ds_test[i]:
        code_space_test.append(saved_model(x["x"], trainable=False)["final_avg_pool"])
        labels_test.append(x["y"])
        l = len(x["y"])
        userid_test.append(l * [id])
code_space_test = np.concatenate(code_space_test)
labels_test = np.concatenate(labels_test)
userid_test = np.concatenate(userid_test)

labels = np.concatenate([labels_train, labels_test])
code_space = np.concatenate([code_space_train, code_space_test])
userid = np.concatenate([userid_train, userid_test])

np.save(os.path.join(SAVED_EMBEDDINGS_PTH, SAVED_EMBEDDINGS_FILENAME), code_space)
np.save(os.path.join(SAVED_EMBEDDINGS_PTH, SAVED_LABELS_FILENAME), labels)
np.save(os.path.join(SAVED_EMBEDDINGS_PTH, SAVED_USERID_FILENAME), userid)

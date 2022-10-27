#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from simclr_preprocessing import preprocess_image

tf.compat.v1.enable_v2_behavior()
print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # You can activate GPU here

SAVED_EMBEDDINGS_PTH = "./"
SAVED_EMBEDDINGS_FILENAME = "code_space.npy"
SAVED_LABELS_FILENAME = "labels.npy"


####################
### 1. SimCLR Embedding Extraction
####################

DATASET_NAME = "cifar10"
BATCH_SIZE = 8  # for SimCLR prediction only

[tfds_dataset_train, tfds_dataset_test], tfds_info = tfds.load(
    DATASET_NAME, split=["train", "test"], with_info=True
)


def _preprocess(img):
    img["image"] = preprocess_image(
        img["image"], 224, 224, is_training=False, test_crop=True
    )
    return img


ds_train = tfds_dataset_train.map(_preprocess).batch(BATCH_SIZE)
ds_test = tfds_dataset_test.map(_preprocess).batch(BATCH_SIZE)

SAVED_MODEL_PTH = (
    "gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r152_3x_sk1/saved_model/"
)
saved_model = tf.saved_model.load(SAVED_MODEL_PTH)


code_space_train, code_space_test = [], []
labels_train, labels_test = [], []

for x in ds_train:
    code_space_train.append(saved_model(x["image"], trainable=False)["final_avg_pool"])
    labels_train.append(x["label"])
code_space_train = np.concatenate(code_space_train)
labels_train = np.concatenate(labels_train)

for x in ds_test:
    code_space_test.append(saved_model(x["image"], trainable=False)["final_avg_pool"])
    labels_test.append(x["label"])
code_space_test = np.concatenate(code_space_test)
labels_test = np.concatenate(labels_test)

labels = np.concatenate([labels_train, labels_test])
code_space = np.concatenate([code_space_train, code_space_test])

np.save(os.path.join(SAVED_EMBEDDINGS_PTH, SAVED_EMBEDDINGS_FILENAME), code_space)
np.save(os.path.join(SAVED_EMBEDDINGS_PTH, SAVED_LABELS_FILENAME), labels)

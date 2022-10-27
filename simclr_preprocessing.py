import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()

#####
## This license information is only for this file ##
## license: Apache-2.0 ##
## authors: google-research/simclr, Google 2020. ##
## URL: https://github.com/google-research/simclr/blob/master/tf2/colabs/load_and_inference.ipynb ##
#####

CROP_PROPORTION = 0.875  # Standard for ImageNet.


def _compute_crop_shape(
        image_height, image_width, aspect_ratio, crop_proportion):
    """Compute aspect ratio-preserving shape for central crop.
    The resulting shape retains `crop_proportion` along one side and a proportion
    less than or equal to `crop_proportion` along the other side.
    Args:
      image_height: Height of image to be cropped.
      image_width: Width of image to be cropped.
      aspect_ratio: Desired aspect ratio (width / height) of output.
      crop_proportion: Proportion of image to retain along the less-cropped side.
    Returns:
      crop_height: Height of image after cropping.
      crop_width: Width of image after cropping.
    """
    image_width_float = tf.cast(image_width, tf.float32)
    image_height_float = tf.cast(image_height, tf.float32)

    def _requested_aspect_ratio_wider_than_image():
        crop_height = tf.cast(tf.math.rint(
            crop_proportion / aspect_ratio * image_width_float), tf.int32)
        crop_width = tf.cast(tf.math.rint(
            crop_proportion * image_width_float), tf.int32)
        return crop_height, crop_width

    def _image_wider_than_requested_aspect_ratio():
        crop_height = tf.cast(
            tf.math.rint(crop_proportion * image_height_float), tf.int32)
        crop_width = tf.cast(tf.math.rint(
            crop_proportion * aspect_ratio *
            image_height_float), tf.int32)
        return crop_height, crop_width

    return tf.cond(
        aspect_ratio > image_width_float / image_height_float,
        _requested_aspect_ratio_wider_than_image,
        _image_wider_than_requested_aspect_ratio)


def center_crop(image, height, width, crop_proportion):
    """Crops to center of image and rescales to desired size.
    Args:
      image: Image Tensor to crop.
      height: Height of image to be cropped.
      width: Width of image to be cropped.
      crop_proportion: Proportion of image to retain along the less-cropped side.
    Returns:
      A `height` x `width` x channels Tensor holding a central crop of `image`.
    """
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    crop_height, crop_width = _compute_crop_shape(
        image_height, image_width, height / width, crop_proportion)
    offset_height = ((image_height - crop_height) + 1) // 2
    offset_width = ((image_width - crop_width) + 1) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width)

    image = tf.compat.v1.image.resize_bicubic([image], [height, width])[0]

    return image


def preprocess_for_eval(image, height, width, crop=True):
    """Preprocesses the given image for evaluation.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      crop: Whether or not to (center) crop the test images.
    Returns:
      A preprocessed image `Tensor`.
    """
    if crop:
        image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image

def preprocess_image(image, height, width, is_training=False,
                     color_distort=True, test_crop=True):
    """Preprocesses the given image.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      is_training: `bool` for whether the preprocessing is for training.
      color_distort: whether to apply the color distortion.
      test_crop: whether or not to extract a central crop of the images
          (as for standard ImageNet evaluation) during the evaluation.
    Returns:
      A preprocessed image `Tensor` of range [0, 1].
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
        return preprocess_for_train(image, height, width, color_distort)
    else:
        return preprocess_for_eval(image, height, width, test_crop)

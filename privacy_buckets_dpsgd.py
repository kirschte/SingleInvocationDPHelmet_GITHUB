# written by David Sommer (david.sommer at inf.ethz.ch)
# original template: https://github.com/sommerda/privacybuckets/blob/master/example_composition.py
# modified by Authors of "Single SMPC Invocation DPHelmet: Differentially Private Distributed Learning on a Large Scale"

import traceback
from scipy.stats import norm
import numpy as np
from privacybuckets.core.probabilitybuckets_light import ProbabilityBuckets

####
# Privacy Buckets is based on the following publication
####

# [*] S. Meiser, E. Mohammadi,
# [*] "Tight on Budget? Tight Bounds for r-Fold Approximate Differential Privacy",
# [*] Proceedings of the 25th ACM Conference on Computer and Communications Security (CCS), 2018

# We examine subsampled Gaussian distributions defined as follows
noise_multiplier = 16
DELTA_TARGET = 1e-5  # the target delta for a given noise multiplier
x_value = np.linspace(-40*noise_multiplier, 40*noise_multiplier, num=5000000, endpoint=True)

Q = 1024/50000  # subsampling rate
EPOCHS = 40
n_iter = int(EPOCHS * np.floor(1 / Q))
SIGMA = noise_multiplier


composition_goal = int(n_iter)
print('composition_goal', composition_goal)


# Important: the individual input distributions neede each to sum up to 1 exactly!
distribution_A = norm.pdf(x_value, loc=0, scale=SIGMA)
distribution_A = distribution_A/np.sum(distribution_A)
distribution_B = norm.pdf(x_value, loc=1, scale=SIGMA)
distribution_B = distribution_B/np.sum(distribution_B)
distribution_B = Q*distribution_B + (1-Q)*distribution_A


### Choosing the number of buckets.
#
# We distribute the elements of distribution_A in the buckets according to the privacy loss L_A/B.
# Higher number-of_buckets allows more finegrade composition. The runtime if composition is quadratic in the number of buckets.
# A good value to start with is
number_of_buckets = 100000
# It is more beneficial to adapt the factor (see below) than the number-of-buckets.
# Once you understand how it influcences the accuracy, change it to any value you like and your hardware supports.


### Choosing the factor
#
# The factor is the average additive distance between the privacy-losses that are put in neighbouring buckets.
# You want to put most of you probablity mass of distribution_A in the buckets, as the rest gets put into the
# infitity_bucket and the minus_n-bucket. Therefore for an o
#
#       L_A/B(o) = log(factor) * number_of_buckets / 2
#
# then the mass Pr[ L_A/B(o) > mass_infinity_bucket, o <- distribution_A] will be put into the infinity bucket.
# The infinity-bucket gros exponentially with the number of compositions. Chose the factor according to the
# probability mass you want to tolerate in the inifity bucket. For this example, the minimal factor should be
#
#       log(factor) > eps
#
# as for randomized response, there is no privacy loss L_A/B greater than epsilon (excluding delta/infinity-bucket).
# We set the factor to
factor = 1 + 1e-6


# Initialize privacy buckets.
privacybuckets = ProbabilityBuckets(
        number_of_buckets=number_of_buckets,
        factor=factor,
        dist1_array=distribution_B,  # distribution B
        dist2_array=distribution_A,  # distribution A
        caching_directory="./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10**(-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
        )

# Now we evaluate how the distributon looks after `composition_goal` independent compositions
privacybuckets_composed = privacybuckets.compose(composition_goal)


# Print status summary
privacybuckets_composed.print_state()

print(f'eps for delta={DELTA_TARGET} is {privacybuckets_composed.eps_ADP_upper_bound(DELTA_TARGET)}')

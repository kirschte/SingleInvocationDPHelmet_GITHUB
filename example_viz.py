#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(rc={"figure.figsize": (11.7, 8.27)})  # larger figures

# read experiments data here
tests_dphelmet = pd.read_csv("tests_dphelmet.csv")


tests_sampl = pd.concat(
    [
        tests_dphelmet,
        # include more data if needed
    ],
    ignore_index=True,
)

# determine the best (= highest mean test accuracy) l2 and Lambda parameters per eps, n_users, and n_per_user
tests_sampl = (
    tests_sampl.groupby(["dp_eps", "n_users", "n_per_user"])
    .apply(
        lambda x: x[
            (
                x["lambda"]
                == x.groupby(["dp_eps", "lambda", "l2", "n_users", "n_per_user"])
                .mean()
                .reset_index()
                .loc[
                    x.groupby(["dp_eps", "lambda", "l2", "n_users", "n_per_user"])
                    .mean()
                    .reset_index()
                    .idxmax(axis=0)["test_acc"]
                ]["lambda"]
            )
            & (
                x["l2"]
                == x.groupby(["dp_eps", "lambda", "l2", "n_users", "n_per_user"])
                .mean()
                .reset_index()
                .loc[
                    x.groupby(["dp_eps", "lambda", "l2", "n_users", "n_per_user"])
                    .mean()
                    .reset_index()
                    .idxmax(axis=0)["test_acc"]
                ]["l2"]
            )
        ]
    )
    .reset_index(drop=True)
)

# apply the eps-correction (for delta=1e-5) from the noise estimate via privacy buckets
# (this has to be modified for other eps values or a different delta or dataset)
# PrivacyBuckets: https://github.com/sommerda/privacybuckets
tests_sampl = tests_sampl.replace(
    {
        "dp_eps": {
            0.1: 0.06071,
            0.2: 0.1299,
            0.5: 0.3526,
            0.8: 0.5885,
            1.0: 0.7511,
            1.5: 1.172,
            2.0: 1.611,
            5.0: 4.541,
            10.0: 10.40,
            100: 300.2,
            # for delta=2e-8 on the same noise scale estimate (i.e. do not change the DELTA variable)
            # 0.1: 0.09167, 0.2: 0.1896, 0.5: 0.4957, 0.8: 0.8128, 1.0: 1.029,
            # 1.5: 1.582, 2.0: 2.152, 5.0: 5.854, 10.0: 12.98, 100.0: 325.5,
        }
    }
)

# renaming for better plotting
tests_sampl = tests_sampl.replace({"variant": {"dist_dphelmet": "DP-SGD-SVM"}})
tests_sampl = tests_sampl.rename(
    columns={"n_users": "$\\#$users", "n_per_user": "$\\#$data points per user"}
)


#####
### plot Figure 4 (right) ####
#####


ax = sns.lineplot(
    data=tests_sampl[tests_sampl["$\\#$users"] != 1],
    x="dp_eps",
    y="test_acc",
    hue="variant",
    style="$\\#$users",
    ci="sd",
    palette=sns.color_palette()[: len(tests_sampl["variant"].value_counts())],
)

plt.ylim([0.1, 0.97])
plt.xlim([0.09, 5.25])
plt.xscale("log")
ax.set_xticks([0.1, 0.3, 1.0, 3.0])
plt.ylabel("test accuracy")
plt.xlabel("$\\varepsilon$ (privacy budget)")
sns.despine()
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.get_major_formatter().set_scientific(False)
plt.grid(b=True, which="minor", color="w", linestyle="dotted")

plt.savefig("figure4_right.png", bbox_inches="tight", pad_inches=0)
plt.close()


#####
#### plot Figure 4 (left) ####
#####


tests_sampl_2 = tests_sampl[
    tests_sampl["$\\#$data points per user"] * tests_sampl["$\\#$users"] > 49999
]

ax = sns.lineplot(
    data=tests_sampl_2,
    x="dp_eps",
    y="test_acc",
    hue="variant",
    style="$\\#$users",
    ci="sd",
    palette=sns.color_palette()[: len(tests_sampl_2["variant"].value_counts())],
)

plt.ylim([0.27, 0.97])
plt.xlim([0.09, 5.25])
plt.xscale("log")
ax.set_xticks([0.1, 0.3, 1.0, 3.0])
plt.ylabel("test accuracy")
plt.xlabel("$\\varepsilon$ (privacy budget)")
sns.despine()
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.get_major_formatter().set_scientific(False)
plt.grid(b=True, which="minor", color="w", linestyle="dotted")

plt.savefig("figure4_left.png", bbox_inches="tight", pad_inches=0)
plt.close()


#####
#### plot Figure 5 ####
#####


TARGET_EPS = 0.5885
tests_sampl_3 = tests_sampl[
    (tests_sampl["dp_eps"] == TARGET_EPS)
    & (tests_sampl["$\\#$data points per user"] == 50)
]


ax = sns.lineplot(
    data=tests_sampl_3,
    x="$\\#$users",
    y="test_acc",
    hue="variant",
    ci="sd",
    palette=sns.color_palette()[: len(tests_sampl_3["variant"].value_counts())],
)

plt.ylim([0.13, 0.83])
plt.xlim([100, 1000])
plt.ylabel("test accuracy")
sns.despine()
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.get_major_formatter().set_scientific(False)
plt.grid(b=True, which="minor", color="w", linestyle="dotted")

plt.savefig("figure5.png", bbox_inches="tight", pad_inches=0)
plt.close()


#####
#### plot Figure 3 (left) ####
#####
# Note: Only the case without Corollary 14 is plotted here
# For Corollary 14 you have to make a rerun of the program with the changed sensitivity.


tests_sampl_4 = tests_sampl[
    (tests_sampl["variant"] == "DP-SGD-SVM")
    & (tests_sampl["$\\#$data points per user"] == 50)
    & (tests_sampl["$\\#$users"] == 1000)
]

user_factor = 200  # for 10M data points
tests_result = pd.DataFrame(columns=None)
e = np.asrray([0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
for k in np.concatenate(
    [
        np.linspace(1, 10, 10),
        np.linspace(20, 50, 4),
    ]
):
    tmp = tests_sampl_4.groupby("dp_eps").mean()["test_acc"]
    tmp.index = tmp.index / user_factor * k
    tmp = tmp.append(pd.Series([np.nan] * len(e), index=e)).sort_index().interpolate()
    tests_result = tests_result.append(
        [{"k": int(k), "dp_eps": np.around(e_, 6), "test_acc": tmp[e_]} for e_ in e],
        ignore_index=True,
    )

tests_result = tests_result.pivot_table(columns="dp_eps", index="k", values="test_acc")

ax = sns.heatmap(
    data=tests_result.iloc[[0, 1, 13], :],
    square=True,
    vmin=0.25,
    vmax=0.9,
    annot=True,
    annot_kws={"size": 13},
    cbar=False,
)
sns.despine()
ax.invert_yaxis()
plt.ylabel("$\\Upsilon$ groups")
plt.yticks(rotation=0)
plt.xlabel("$\\varepsilon$ (privacy budget)")

plt.savefig("figure3_left.png", bbox_inches="tight", pad_inches=0)
plt.close()


#####
#### plot Figure 3 (right) ####
#####
# Note: Only the case without Corollary 14 is plotted here
# For Corollary 14 you have to make a rerun of the program with the changed sensitivity.

tests_sampl_4 = tests_sampl[
    (tests_sampl["variant"] == "DP-SGD-SVM")
    & (tests_sampl["$\\#$data points per user"] == 50)
    & (tests_sampl["$\\#$users"] == 1000)
]

user_factor = 20000  # for 1B data points
tests_result = pd.DataFrame(columns=None)
e = np.asarray([0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005])
for k in np.concatenate(
    [
        np.linspace(1, 10, 10),
        np.linspace(20, 50, 4),
    ]
):
    tmp = tests_sampl_4.groupby("dp_eps").mean()["test_acc"]
    tmp.index = tmp.index / user_factor * k
    tmp = tmp.append(pd.Series([np.nan] * len(e), index=e)).sort_index().interpolate()
    tests_result = tests_result.append(
        [{"k": int(k), "dp_eps": np.around(e_, 6), "test_acc": tmp[e_]} for e_ in e],
        ignore_index=True,
    )

tests_result = tests_result.pivot_table(columns="dp_eps", index="k", values="test_acc")

#############

ax = sns.heatmap(
    data=tests_result.iloc[[0, 1, 13], :],
    square=True,
    vmin=0.2,
    vmax=0.9,
    annot=True,
    annot_kws={"size": 13},
    cbar=False,
)
sns.despine()
ax.invert_yaxis()
plt.ylabel("$\\Upsilon$ groups")
plt.yticks(rotation=0)
plt.xlabel("$\\varepsilon$ (privacy budget)")

plt.savefig("figure3_right.png", bbox_inches="tight", pad_inches=0)
plt.close()

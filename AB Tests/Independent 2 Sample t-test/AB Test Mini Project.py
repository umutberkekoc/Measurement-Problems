# Is there a statistically significant difference between the ages of people with and without diabetes?

import pandas as pd
import statsmodels.stats.api as sms
from scipy.stats import (ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,
                         pearsonr, spearmanr, kendalltau, f_oneway, kruskal)

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# 1. Created Hypothesis
#2. Assumption Checking
# - 1. Normality Assumption: The distribution of the relevant groups is Normal distribution
# - 2. Variance Homogeneity: The distribution of variances of two groups is similar
#3. Application of Hypothesis
# - 1. If the assumptions are met, independent two-sample t-test (parametric test) (the number of real lives is less)
# - 2. If the assumptions are not met, mannwhitneyu test (non-parametric test)
# 4. Interpret the results based on p-values.
# NOTE:
# - If normality is not achieved, number 2 directly. If variance homogeneity is not achieved, number 1
# - It may be useful to perform outlier review and correction before normality review.
df = pd.read_csv("diabetes.csv")
df.head()
df.shape
df.columns
df.describe().T
df.isnull().sum()
df.groupby("Outcome").agg({"Age": "mean"})
# Created Hypothesis:
# HO: M1 = M2  (There is not statistically significant difference in the average of age between groups)
# H1: M1 != M2: (There is statistically significant difference in the average of age between groups)

#2. Assumption Checking
# - 1. Normality Assumption: The distribution of the relevant groups is Normal distribution
# HO: Normality Assumption provided
# H1: Normality Assumption not Provided

test_stat, p_value = shapiro(df[df["Outcome"] == 0]["Age"])
print("Shapiro Test:", "Test Stat: {}\tP-Value: {}".format(test_stat, p_value))
test_stat, p_value = shapiro(df[df["Outcome"] == 1]["Age"])
print("Shapiro Test:", "Test Stat: {}\tP-Value: {}".format(test_stat, p_value))
# p-value <0.05, HO Rejected, Normality Assumption not Provided

# 2. Variance Homogeneity: The distribution of variances of two groups is similar
# HO: Homogeneity Variance provided
# H1: Homogeneity Variance not Provided

test_stat, p_value = levene(df.loc[df["Outcome"] == 0, "Age"],
                            df.loc[df["Outcome"] == 1, "Age"])
print("Levene Test:", "Test Stat: {}\tP-Value: {}".format(test_stat, p_value))
# p-value > 0.05, HO Not Rejected, Homogeneity Variance provided

#3. Application of Hypothesis (non-parametric test):
test_stat, p_value = mannwhitneyu(df[df["Outcome"] == 0]["Age"],
                                  df.loc[df["Outcome"] == 1, "Age"])
print("MannWhitneyU Test:", "Test Stat: {}\tP-Value: {}".format(test_stat, p_value))

# 4. Interpret the results based on p-values.
# p-value < 0.05, HO Rejected, Thus, There is  statistically significant difference in the average of age between groups


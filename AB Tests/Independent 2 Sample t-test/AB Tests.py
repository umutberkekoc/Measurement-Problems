import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import statsmodels.stats.api as sms
from scipy.stats import (ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,
                         pearsonr, spearmanr, kendalltau, f_oneway, kruskal)
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)


#####################################
# Sampling (Örnekleme):
population = np.random.randint(0, 80, 10000)  #population created from 0 to 80 with 10.000 elements
population.mean()

sample1 = np.random.choice(population, size=100)  # one sample created from population with 100 elements
sample1.mean()

sample1 = np.random.choice(population, size=100)
sample2 = np.random.choice(population, size=100)
sample3 = np.random.choice(population, size=100)
sample4 = np.random.choice(population, size=100)
sample5 = np.random.choice(population, size=100)
sample6 = np.random.choice(population, size=100)
sample7 = np.random.choice(population, size=100)
sample8 = np.random.choice(population, size=100)
sample9 = np.random.choice(population, size=100)
sample10 = np.random.choice(population, size=100)

large_sample = (sample10 + sample9 + sample8 + sample7 + sample6 + sample4 + sample5 + sample3 + sample2 + sample1) / 10
large_sample.mean()

###########################################
# Descriptive Statistics (Betimsel / Tanımlayıcı İstatistikler):
df = sns.load_dataset("tips")
df.describe().T
df.describe([0.01, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

###########################################
# Confidence Interval (Güven Aralığı):
# CI = X +- Z * S / sqrt(n)
# - X --> Population Mean
# - Z --> CI, %95 --> 1,96 veya %99 --> 2,57
# - S --> Standard deviation
# - n --> Sample size (gözlem sayısı)

sms.DescrStatsW(df["total_bill"]).tconfint_mean()
sms.DescrStatsW(df["tip"]).tconfint_mean()
sms.DescrStatsW(df["size"]).tconfint_mean()

###########################################
# Correlation (Korelasyon):
# It is a statistical method that provides information about the relationship between variables,
# the direction of this relationship and the severity.
# Takes a value between -1 and 1. 1 is perfect positive correlation, -1 is perfect negative correlation, 0 is no correlation. 0.5 low
# Positive correlation as the value of one variable increases, the values of the other variable increase.
# negative correlation As values of one variable increase, values of the other variable decrease.
df["total_bill"] = df["total_bill"] - df["tip"]

sns.scatterplot(data=df, x="tip", y="total_bill")
plt.show()
df["tip"].corr(df["total_bill"])
df["total_bill"].corr(df["tip"])

###########################################
# Hypothesis Testing:
# The main purpose of group comparisons is to try to show whether possible differences arise by chance.
# For example, the packaging of a product changed and the sales rate increased by 3%.
# We can't say it was directly thanks to this packaging! It also happens by chance!


###########################################
# AB Testi (Independent Two Sample T Test):
# It is used when you want to make a comparison between two group averages.
# Ho: X1 = X2
# H1: X1 != X2
# If p value < 0.05 or th > t-table H0 Reject.

# 1. Establish Hypothesis
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



# Application 1: Is there a statistically significant difference between the account averages of smokers and non-smokers?
df = sns.load_dataset("tips")
df.head()
df.groupby("smoker").agg({"total_bill": "mean"})
# H0: M1 = M2
# H1: M1 != M2

#2. Assumption Checking
# - 1. Normality Assumption: The distribution of the relevant groups is Normal distribution
# HO: Normality Assumption provided
# H1: Normality Assumption not Provided

test_stat, p_value = shapiro(df[df["smoker"] == "Yes"]["total_bill"])
print("Normality Assumption:", "test stat: {}\t p-value: {}".format(test_stat, p_value))
test_stat, p_value = shapiro(df[df["smoker"] == "No"]["total_bill"])
print("Normality Assumption:", "test stat: {}\t p-value: {}".format(test_stat, p_value))
# p-value < 0.05, HO Rejected, Normality Assumption provided (Currently we know that we need to use non-parametric test)



# - 2. Variance Homogeneity: The distribution of variances of two groups is similar
# HO: Variance Homogeneity provided
# H1: Variance Homogeneity not Provided
test_stat, p_value = levene(df[df["smoker"] == "Yes"]["total_bill"],
                            df[df["smoker"] == "No"]["total_bill"])
print("Variance Homogeneity:", "test stat: {}\t p-value: {}".format(test_stat, p_value))
# p-value < 0.05, HO Rejected,

#3. Application of Hypothesis and evalulation results
# non-parametric test --> mannwhitneyu test (because, assumptions are not provided)

test_stat, p_value = mannwhitneyu(df[df["smoker"] == "Yes"]["total_bill"],
                                  df[df["smoker"] == "No"]["total_bill"])

print("MannWhitneyU:", "test stat: {}\t p-value: {}".format(test_stat, p_value))
# p-value > 0.05, HO not Rejected
# Thus, There is not statistically significant difference in the average of total bill between groups.


# If assumption checks were provided (normal distribution and variance homogeneity) --> we would use parametric test.
test_stat, p_value = ttest_ind(df[df["smoker"] == "Yes"]["total_bill"],
                               df[df["smoker"] == "No"]["total_bill"],
                               equal_var=True)
print("Independent T Test:", "test stat: {}\t p-value: {}".format(test_stat, p_value))



#####################################################################
# Application 2: Is there a statistically significant difference between the age averages of males and females ?

# HO: M1 = M2  (There is not statistically significant difference in the average of age between genders)
# H1: M1 != M2: (There is statistically significant difference in the average of age between genders)

df = sns.load_dataset("titanic")
df.head()
df.groupby("sex").agg({"age": "mean"})
df.isnull().sum()
# Assumption Checking
# - 1. Normality Assumption: The distribution of the relevant groups is Normal distribution
# HO: Normality Assumption provided
# H1: Normality Assumption not Provided

test_stat, p_value = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print("Normality Assumption:", "test stat: {}\t p-value: {}".format(test_stat, p_value))
test_stat, p_value = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print("Normality Assumption:", "test stat: {}\t p-value: {}".format(test_stat, p_value))
# p-value  0.05, HO Rejected! We do not need to levene test, It's currently known that we will use non-parametric test

# non-parametric test --> mannwhitneyu test:

test_stat, p_value = mannwhitneyu(df[df["sex"] == "female"]["age"].dropna(),
                                  df[df["sex"] == "male"]["age"].dropna())

print("MannWhitneyU:", "test stat: {}\t p-value: {}".format(test_stat, p_value))
# p-value < 0.05, HO Rejected
# Thus, There is statistically significant difference in the average of age between groups(female and male)

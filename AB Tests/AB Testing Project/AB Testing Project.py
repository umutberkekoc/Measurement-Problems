
import pandas as pd
from scipy.stats import (ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,
                         pearsonr, spearmanr, kendalltau, f_oneway, kruskal)
from statsmodels.stats.multicomp import MultiComparison

df_control = pd.read_excel("ab_testing.xlsx", sheet_name="Control Group")
df_test = pd.read_excel("ab_testing.xlsx", sheet_name="Test Group")

def check_dataframe(dataframe):
    print("*** HEAD ***")
    print(dataframe.head())
    print("*** TAIL ***")
    print(dataframe.tail())
    print("*** SHAPE ***")
    print(dataframe.shape)
    print("*** INFO ***")
    print(dataframe.info())
    print("*** DESCRİBE ***")
    print(dataframe.describe().T)
    print("*** # NA ***")
    print(dataframe.isnull().sum())
    print("*** COLUMNS ***")
    print(dataframe.columns)

check_dataframe(df_control)
check_dataframe(df_test)

df = pd.concat([df_control, df_test], ignore_index=True)

df["group"] = ["Control" if i < 40 else "Test" for i in df.index]

print(df.groupby("group").agg({"Purchase": "mean"}))


# HO: M1=M2 (There is not statistically significant difference in means between the groups)
# H1: M1!=M2 (There is a statistically significant difference in means between the groups)

# Normality Test (Checking The Normal Distribution Existing)
test_stat, shap_p_value = shapiro(df[df["group"] == "Control"]["Purchase"])
print("Shapiro--> test stat: {}\tp-value: {}".format(test_stat, shap_p_value))
test_stat, shap_p_value = shapiro(df[df["group"] == "Test"]["Purchase"])
print("Shapiro--> test stat: {}\tp-value: {}".format(test_stat, shap_p_value))

if shap_p_value < 0.05:
    print("HO Rejected: Normality Test is not Provided")
else:
    print("H0 Not Rejected: Normality Test is Provided")

# Homogeneous of Variance Test (Checking The Homogeneity in Variance Existing)
test_stat, p_value = levene(df[df["group"] == "Control"]["Purchase"],
                            df[df["group"] == "Test"]["Purchase"])
print("Levene--> test stat: {}\tp-value: {}".format(test_stat, p_value))

if p_value < 0.05:
    print("HO Rejected: Variances are not Homogeneous")
else:
    print("H0 Not Rejected: Variances are Homogeneous")


if (shap_p_value > 0.05) and (p_value > 0.05):
    test_stat, pvalue = ttest_ind(df[df["group"] == "Control"]["Purchase"],
                                  df[df["group"] == "Test"]["Purchase"],
                                  equal_var=True)
    print("t-test--> test stat: {}\tp-value: {}".format(test_stat, pvalue))

elif (shap_p_value > 0.05) and (p_value < 0.05):
    test_stat, pvalue = ttest_ind(df[df["group"] == "Control"]["Purchase"],
                                  df[df["group"] == "Test"]["Purchase"],
                                  equal_var=False)
    print("t-test--> test stat: {}\tp-value: {}".format(test_stat, pvalue))
else:
    test_stat, pvalue = mannwhitneyu(df[df["group"] == "Control"]["Purchase"],
                                     df[df["group"] == "Test"]["Purchase"])
    print("mannwhitneyu test--> test stat: {}\tp-value: {}".format(test_stat, pvalue))


if pvalue < 0.05:
    print("HO Rejected: There is statistically significant difference in means between the groups")
else:
    print("HO Not Rejected: There is not statistically significant difference in means between the groups")


comparison = MultiComparison(df["Purchase"], df["group"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())
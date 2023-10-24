import pandas as pd
import statsmodels.stats.api as sms
from scipy.stats import (ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,
                         pearsonr, spearmanr, kendalltau, f_oneway, kruskal)

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

#####################################################################
# İş Problemi: Kursun büyük çoğunluğunu izleyenler ile izlemeyenlerin puanları birbirinden farklı mı=
df = pd.read_csv("course_reviews.csv")
df.head()
df.describe().T
df.isnull().sum()
df["progress_situation"] = pd.qcut(df["Progress"], 2, labels=["less_watcher", "more_watcher"])
df.groupby("progress_situation").agg({"Rating": "mean"})

# Hipotezin Kurulması:
# HO: M1 = M2 (kursun büyük çoğunluğunu izleyenler ile izlemeyenlerin puanları aynı, istat.olarak anlamlı fark yok)
# H1: M1 != M2 (anlamlı fark vardır)

# Varsayım Kontrolü:
# - Normallik Varsayımı (normal dağılım olup olmadığı varsayımı)
# - Varyansların homojen olma varsayımı


# Varsayım Kontrolü:
# Normallik Varsayımı
# HO: Normal dağılım varsayımı sağlanmakta
# H1: Normal dağılım varsayımı sağlanmamakta
test_stat, p_value = shapiro(df[df["Progress"] < df["Progress"].describe().T[4]]["Rating"])
print("Shapiro Test:", "Test Stat: {}\tP-Value: {}".format(test_stat, p_value))

test_stat, p_value = shapiro(df[df["Progress"] > df["Progress"].describe().T[6]]["Rating"])
print("Shapiro Test:", "Test Stat: {}\tP-Value: {}".format(test_stat, p_value)) # HO Rejected
# normall dağılım varsayımı gerçekleşemediği için direkt olarak non-parametric test uygulamaya gidebiliriz.

# Varyasların Homojen varsayımı:
# HO: Varyanslar Homojendir
# H1: Varyanslar Homojen değildir
test_stat, p_value = levene(df[df["Progress"] > df["Progress"].describe().T[6]]["Rating"],
                            df[df["Progress"] < df["Progress"].describe().T[4]]["Rating"])
print("Levene Test:", "Test Stat: {}\tP-Value: {}".format(test_stat, p_value))  # p-value<0.05, HO Rejected!


#non-oarametric test (mannwhitneyu test):
test_stat, p_value = mannwhitneyu(df[df["Progress"] > df["Progress"].describe().T[6]]["Rating"],
                                  df[df["Progress"] <  df["Progress"].describe().T[4]]["Rating"])
print("MannWhitneyU Test:", "Test Stat: {}\tP-Value: {}".format(test_stat, p_value))  # p-value <0.05, HO Rejected!

# 4. Interpret the results based on p-values.
# p-value < 0.05, HO Rejected,
# Sonuç, kursun büyük çoğunluğunu izleyenler ile izlemeyenlerin puanları arasında istatiksel olarak anlamlı fark vardır!

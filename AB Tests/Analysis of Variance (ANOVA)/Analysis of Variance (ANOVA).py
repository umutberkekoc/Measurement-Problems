import pandas as pd
import seaborn as sns
import itertools
import statsmodels.stats.api as sms
from scipy.stats import (ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,
                         pearsonr, spearmanr, kendalltau, f_oneway, kruskal)
from statsmodels.stats.multicomp import MultiComparison

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# İkiden fazla grup ortalaması karşılaştırma (ANOVA- Analysis of Variance)

# HO : M1 = M2 = M3
# H1: Eşit değillerdir, en az birisi farklıdır
df= sns.load_dataset("tips")
df.head()
df.isnull().sum()
df.shape


df.groupby("day").agg({"total_bill": "mean"})

# Hipotezlerin Kurulması:
# HO: m1 = m2 = m3 = m4 (grup ortalamaları arasında fark yoktur)
# H1: fark vardır

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa --> one way anove
# varsayım sağlanmıyorsa --> kruskal

# Normallik varsayımı:
# H0: Normal dağılım varsayımı sağlanmaktadır
# H1: Normal dağılım varsayımı sağlanmamaktadır

# Varyans homojenliği varsayımı
# HO: Varyans Homojenliği sağlanmaktadır
# H1: Varyans Homojenliği Sağlanmamaktadır


# Normallik Varsayımı:
# HO: Normal dağılım varsayımı geçerlidir
# H1: Normal dağılım varsayımı geçerli değildir

test_stat, p_value = shapiro(df[df["day"] == "Thur"]["total_bill"])
print("test stat1: {}\tp-value1: {}".format(test_stat, p_value))
test_stat, p_value = shapiro(df[df["day"] == "Fri"]["total_bill"])
print("test stat2: {}\tp-value2: {}".format(test_stat, p_value))
test_stat, p_value = shapiro(df[df["day"] == "Sat"]["total_bill"])
print("test stat3: {}\tp-value3: {}".format(test_stat, p_value))
test_stat, p_value = shapiro(df[df["day"] == "Sun"]["total_bill"])
print("test stat4: {}\tp-value4: {}".format(test_stat, p_value))


# shorter way!
for i in list(df["day"].unique()):
    test_stat, p_value = shapiro(df[df["day"] == i]["total_bill"])
    print("test stat: {}\tp-value: {}".format(test_stat, p_value)) # p-value < 0.05, HO Rejected!
# Thus, H1: Normal dağılım varsayımı sağlanmamaktadır


# Varyans homojenliği varsayımı

test_stat, p_value = levene(df[df["day"] == "Thur"]["total_bill"],
                            df[df["day"] == "Fri"]["total_bill"],
                            df[df["day"] == "Sat"]["total_bill"],
                            df[df["day"] == "Sun"]["total_bill"])


# Hipotez testi ve p-value yorumu:
# HO: m1 = m2 = m3 = m4 (grup ortalamaları arasında fark yoktur)
# H1: fark vardır (grup ortalamaları arasında anlamsal fark vardır)


# Normallik varsayımı sağlanamadığı için non-parametrik test kullaıyoruz:
kruskal(df[df["day"] == "Thur"]["total_bill"],
                             df[df["day"] == "Fri"]["total_bill"],
                             df[df["day"] == "Sat"]["total_bill"],
                             df[df["day"] == "Sun"]["total_bill"])  # p-value < 0.05
# Thus, HO: Rejected, There is statistically significant difference between the 4 groups in their average of total bill

# Eğer normallik varsayımı karşılanmış olsaydı (HO NOT REJECT) -->
f_oneway(df[df["day"] == "Thur"]["total_bill"],
                             df[df["day"] == "Fri"]["total_bill"],
                             df[df["day"] == "Sat"]["total_bill"],
                             df[df["day"] == "Sun"]["total_bill"])


# Nereden kaynaklı bu farklılık? ikili p-value karşılaştırma yapabiliriz.
from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["total_bill"], df["day"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# H0: p1 = p2
# Yeni tasarımın dönüşüm oranı ile eski tasarımın dönüşüm oranı arasında istatiskel olarak anlamlı farklılık yoktur
# H1: p1 != p2
# Yeni tasarımın dönüşüm oranı ile eski tasarımın dönüşüm oranı arasında istatiskel olarak anlamlı farklılık vardır

basari_sayisi = np.array([300, 250])
gozlem_sayisi = np.array([1000, 1100])

basari_sayisi / (gozlem_sayisi)

proportions_ztest(nobs=gozlem_sayisi, count=basari_sayisi)

##########################################################
# Uygulama: Kadın ve erkeklerin hayatta kalma oranları arasında istatiksel olarak anlamlı farklılık var mıdır?
# HO: p1 = p2
# Kadın ve erkeklerin hayatta kalma oranları arasında istat. olarak anlamlı farklılık yoktur.
# H1: p1 ! = p2
# Kadın ve erkeklerin hayatta kalma oranları arasında istat. olarak anlamlı farklılık vardır.

df = sns.load_dataset("titanic")
df.head()
df.isnull().sum()
df.groupby("sex").agg({"survived": "sum"})
kadin_basari = df[df["sex"] == "female"]["survived"].sum()
erkek_basari = df[df["sex"] == "male"]["survived"].sum()

kadin_gozlem_sayisi = df[df["sex"] == "female"]["survived"].count()
erkek_gozlem_sayisi = df[df["sex"] == "male"].shape[0]

test_stat, p_value=proportions_ztest(count=[kadin_basari, erkek_basari],
                  nobs=[kadin_gozlem_sayisi, erkek_gozlem_sayisi])
print("test stat: {}\tp-value: {}".format(test_stat, p_value))
# p-value<0.05, HO Rejected!
# Thus, #Kadın ve erkeklerin hayatta kalma oranları arasında istatiksel olarak anlamlı farklılık vardır.


######################################################
# class1 ve class2 hayatta kalma oranları arasında istatiksel olarak anlamlı farklılık var mıdır?
# HO: p1 = p2
# class1 ve class2 hayatta kalma oranları arasında istatiksel olarak anlamlı farklılık yoktur.
# H1: p1 ! = p2
# class1 ve class2 hayatta kalma oranları arasında istatiksel olarak anlamlı farklılık vardır.

df.groupby("class").agg({"survived":"mean"})

first_class_basari = df[df["class"] == "First"]["survived"].sum()
second_class_basari = df[df["class"] == "Second"]["survived"].sum()

first_class_nobs = df.loc[df["class"] == "First"].shape[0]
second_class_nobs = df.loc[df["class"] == "Second"].shape[0]

test_stat, p_value = proportions_ztest(count=[first_class_basari, second_class_basari],
                                       nobs=[first_class_nobs, second_class_nobs])
print("test stat: {}\tp-value: {}".format(test_stat, p_value))
# p-value < 0.05, class1 ve class2 hayatta kalma oranları arasında istatiksel olarak anlamlı farklılık vardır.



# 2 Sample Ratio Test Fonksiyonlaştırma:

def two_sample_ratio_test(dataframe, cat_var, num_var, plot=False):
    first_group_value = input("enter first group value")
    second_group_value = input("enter first group value")
    if (first_group_value not in dataframe[cat_var].values) or (second_group_value not in dataframe[cat_var].values):
        print("Sınıf Değerlerini Doğru Giriniz!")
        return None
    else:
        success_group1 = dataframe[dataframe[cat_var] == first_group_value][num_var].sum()
        success_group2 = dataframe[dataframe[cat_var] == second_group_value][num_var].sum()
        nobs_group1 = dataframe[dataframe[cat_var] == first_group_value].shape[0]
        nobs_group2 = dataframe[dataframe[cat_var] == second_group_value].shape[0]
        test_stat, p_value = proportions_ztest(count=[success_group1, success_group2],
                                               nobs=[nobs_group1, nobs_group2])
    if plot == True:
        sns.barplot(data=df, x=cat_var, y=num_var, estimator="mean", palette="viridis")
        plt.title(label=("Average", num_var, "By", cat_var))
        plt.xlabel(cat_var)
        plt.ylabel(num_var)
        plt.grid()
        plt.xticks(rotation=45)
        plt.show()
    print(dataframe.groupby(cat_var).agg({num_var: "mean"}))
    print("test stat: {}\tp-value: {}".format(test_stat, p_value))
    return test_stat, p_value
if p_value < 0.05:
    print("HO is Rejected. There is a statistically significant difference between two groups in their ratio!")
else:
    print("HO is not Rejected. There is not statistically significant difference between two groups in their ratio!")

two_sample_ratio_test(df, cat_var="sex", num_var="survived", plot=True)


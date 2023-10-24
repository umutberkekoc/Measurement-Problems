
import pandas as pd
import seaborn as sns
import itertools
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

#*****************************************************************
# TÜM İŞLEMLERİ FONKSİYONLAŞTIRMA:::
#*****************************************************************
def anova(dataframe, cat_var, num_var, CI= 1 - float(input("enter confidence interval"))):
    'Normallik Varsayımı:'
    for i in list(dataframe[cat_var].unique()):
        test_stat, p_value = shapiro(dataframe[dataframe[cat_var] == i][num_var])
        print("test stat: {}\tp-value: {}".format(test_stat, p_value))

        if p_value < 0.05:
            print("Shapiro Testi: H0 Reddedildi!, Normallik dağılım varsayımı karşılanamadı")
            ask = input("Yine de Varyans Homojenliği Varsayımı Testini Yapmak İstiyor musunuz? / yes or no")
            if ask == "yes":
                day_unique_variables = dataframe[cat_var].unique()

                # Ask the user for the number of groups (combinations) they want
                num_groups = int(input("Enter the number of groups (combinations): "))

                if num_groups != len(day_unique_variables):
                    print("Invalid number of groups. Please choose a number between 2 and", len(day_unique_variables))
                    break
                else:
                    # Initialize an empty list to store the p-values
                    p_values = []

                    # Use itertools combinations to get all unique variable pairs
                    combinations = itertools.combinations(day_unique_variables, num_groups)

                    ' Varyans Homojenliği Varsayımı'

                    for i in combinations:
                        group_data = [dataframe[dataframe[cat_var] == day][num_var] for day in i]
                        test_stat, p_value = levene(*group_data)
                        p_values.append((i, p_value))

                    # Check if any Levene p-value is less than a significance level (e.g., 0.05)
                    significance_level = 0.05
                    if any(p_value < significance_level for combination, p_value in p_values):
                        print("Levene Testi: H0: Reddedildi, Varyanslar Homojen Değil")
                    else:
                        print("Levene Testi: H0: Kabul Edildi, Varyanslar Homojen")

                    'non-parametric test uygulanımı'
                    # Initialize a dictionary to store the group data
                    group_data = {}

                    # Collect data for the specified number of groups
                    for i, day in enumerate(day_unique_variables[:num_groups]):
                        group_data[f'group_{i + 1}'] = dataframe[dataframe[cat_var] == day][num_var]

                    # Perform Kruskal-Wallis test
                    test_stat, p_value = kruskal(*group_data.values())

                    print(f"Kruskal-Wallis Test Statistic: {test_stat}")
                    print(f"P-Value: {p_value}")
                    if p_value < 0.05:
                        print("H0: Reddedildi, Gruplar Arasında İstatiksel olarak anlamlı fark vardır")
                        break
                    else:
                        print("H0: Kabul Edildi, Gruplar Arasında İstatiksel olarak anlamlı fark yoktur")
                        break
            else:
                'non-parametric test uygulanımı'
                day_unique_variables = dataframe[cat_var].unique()
                num_groups = int(input("Enter the number of groups (combinations): "))
                group_data = {}

                # Collect data for the specified number of groups
                for i, day in enumerate(day_unique_variables[:num_groups]):
                    group_data[f'group_{i + 1}'] = dataframe[dataframe[cat_var] == day][num_var]

                # Perform Kruskal-Wallis test
                test_stat, p_value = kruskal(*group_data.values())

                print(f"Kruskal-Wallis Test Statistic: {test_stat}")
                print(f"P-Value: {p_value}")
                if p_value < 0.05:
                    print("H0: Reddedildi, Gruplar Arasında İstatiksel olarak anlamlı fark vardır")
                    break
                else:
                    print("H0: Kabul Edildi, Gruplar Arasında İstatiksel olarak anlamlı fark yoktur")
                    break


        else:
            print("Shapiro Testi: H0 Reddedilemedi!, Normallik dağılım varsayımı karşılandı")
            day_unique_variables = dataframe[cat_var].unique()
            num_groups = int(input("Enter the number of groups (combinations): "))
            if num_groups != len(day_unique_variables):
                print("Invalid number of groups. Please choose a number between 2 and", len(day_unique_variables))
                break

            ' Varyans Homojenliği Varsayımı'
            p_values = []
            day_unique_variables = dataframe[cat_var].unique()
            num_groups = int(input("Enter the number of groups (combinations): "))
            # Use itertools combinations to get all unique variable pairs
            combinations = itertools.combinations(day_unique_variables, num_groups)

            for i in combinations:
                group_data = [dataframe[dataframe[cat_var] == day][num_var] for day in i]
                test_stat, p_value = levene(*group_data)
                p_values.append((i, p_value))

            # Check if any Levene p-value is less than a significance level (e.g., 0.05)
            significance_level = 0.05
            if any(p_value < significance_level for combination, p_value in p_values):
                print("Levene Testi: H0: Reddedildi, Varyanslar Homojen Değil")
            else:
                print("Levene Testi: H0: Kabul Edildi, Varyanslar Homojen")

            'parametric test uygulanımı'
            day_unique_variables = dataframe[cat_var].unique()
            num_groups = int(input("Enter the number of groups (combinations): "))
            group_data = {}

            # Collect data for the specified number of groups
            for i, day in enumerate(day_unique_variables[:num_groups]):
                group_data[f'group_{i + 1}'] = dataframe[dataframe[cat_var] == day][num_var]

            # Perform Kruskal-Wallis test
            test_stat, p_value = f_oneway(*group_data.values())

            print(f"f_oneway Test Statistic: {test_stat}")
            print(f"P-Value: {p_value}")
            if p_value < 0.05:
                print("H0: Reddedildi, Gruplar Arasında İstatiksel olarak anlamlı fark vardır")
                break
            else:
                print("H0: Kabul Edildi, Gruplar Arasında İstatiksel olarak anlamlı fark yoktur")
                break

anova(df,  cat_var="day",  num_var="total_bill")





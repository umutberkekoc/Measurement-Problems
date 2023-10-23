# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################
import matplotlib.pyplot as plt
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı

import pandas as pd
import numpy as np
import math
import seaborn as sns
import datetime as dt
import scipy.stats as st
import matplotlib.pyplot as plt

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)
df = pd.read_csv("amazon_review.csv")

def show_info(dataframe):
    print("*** HEAD ***")
    print(df.head())
    print("*** TAIL ***")
    print(df.tail())
    print("*** SHAPE ***")
    print(df.shape)
    print("*** INFO ***")
    print(df.info())
    print("*** COLUMNS ***")
    print(df.columns)
    print("*** NA ***")
    print(df.isnull().sum())
    print("*** DESCRIPTION ***")
    print(df.describe([0.01, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)
    print("*** UNIQUE ***")
    print(df.nunique())

show_info(df)
# Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız. (time-based weighted average)
# tarihleri çeyreklik değerlere göre 4 farklı parçaya göre belirleyiniz (0,25,50,75,100)
# katsayıları kullanıcıya girdiriniz.
# Daha sonra bu işlemi her veri seti için fonksiyonlaştırınız

def time_based_weighted_average_calculator(dataframe, variable="day_diff", variable2="overall"):
    w1 = int(input("w1"))
    w2 = int(input("w2"))
    w3 = int(input("w3"))
    w4 = int(input("w4"))

    if w1 + w2 + w3 + w4 != 100:
        raise ValueError("Ağırlık derecelerini tekrar doğru bir şekilde Giriniz!".upper())
    else:
        time_based_average_rating = dataframe[dataframe[variable] <= dataframe[variable].describe().T[4]][variable2].mean() * w1 / 100 + \
            dataframe[(dataframe[variable] > dataframe[variable].describe().T[4]) & (dataframe[variable] <= dataframe[variable].describe().T[5])][variable2].mean() * w2 / 100 + \
            dataframe[(dataframe[variable] > dataframe[variable].describe().T[5]) & (dataframe[variable] <= dataframe[variable].describe().T[6])][variable2].mean() * w3 / 100 + \
            dataframe[dataframe[variable] > dataframe[variable].describe().T[6]][variable2].mean() * w4 / 100
        return time_based_average_rating

try:
    result = time_based_weighted_average_calculator(df)
    print("Time Based Weighted Average: {}".format(result))
    print("General Mean: {}".format(df["overall"].mean()))
except ValueError as e:
    print(e)

# Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
q01 = df[df["day_diff"] <= df["day_diff"].describe().T[4]]["overall"].mean() * 22 / 100  # % 25 den düşük olanlar
q12 = df[(df["day_diff"] > df["day_diff"].describe().T[4]) & (df["day_diff"] <= df["day_diff"].describe().T[5])]["overall"].mean() * 24/100 # 25 ile %50 çeyreklik değer arasındakiler
q23 = df[(df["day_diff"] > df["day_diff"].describe().T[5]) & (df["day_diff"] <= df["day_diff"].describe().T[6])]["overall"].mean() * 26/100 # 50 -75 çeyreklik değer arasındakiler.
q34 = df[df["day_diff"] > df["day_diff"].describe().T[6]]["overall"].mean() * 28/100 # 75 ile 100 çeyreklik arası

list = [q01, q12, q23, q34]
for i in list:
    print(i)

sns.barplot(data=df, y=list, x=["q01", "q12", "q23", "q34"], estimator="sum")
plt.title("Her Zaman Dilimindeki Ağırlıklı Ortalama")
plt.xlabel("Zaman Dilimleri")
plt.ylabel("Ağırlıklı Ortalama")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())

# 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
# score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()


# up-down difference score:
def up_down_diff_score(up, down):
    if up  + down == 0:
        return 0
    else:
        return up - down


# up ratio score:
def average_up_rating_score(up, down):
    if up + down == 0:
        return 0
    else:
        return up / (up+down)


# wilson lower bound score:
def wilson_lower_bound(up, down, confidence=float(input("Enter Confidence Level, 0.99,0.95,....."))):
    """
        Calculate the Wilson Lower Bound Score

        The lower bound of the confidence interval to be calculated for the Bernoulli parameter 'p' is considered the WLB score.
        This score is used for product ranking.
        Note:
        If the scores are in the range of 1-5, they are categorized as 1-3 negative and 4-5 positive and can be adapted for Bernoulli.
        However, this approach brings certain challenges, hence the need for Bayesian average rating.

        Parameters
        ----------
        up: int
            Count of positive outcomes
        down: int
            Count of negative outcomes
        confidence: float
            Confidence level

        Returns
        -------
        wilson score: float
        """

    n = up + down
    if n == 0:
        return 0
    else:
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        phat = up / n
        return (phat + z ** 2 / (2 * n) - z * math.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)) / (1 + z ** 2 / n)

df["score_pos_neg_diff"] = df.apply(lambda x: up_down_diff_score(x["helpful_yes"], x["helpful_no"]), axis=1)
print(df.sort_values("score_pos_neg_diff", ascending=False).head(40))

df["score_average_rating"] = df.apply(lambda x: average_up_rating_score(x["helpful_yes"], x["helpful_no"]), axis=1)
print(df.sort_values("score_average_rating", ascending=False).head(40))

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
print(df.sort_values("wilson_lower_bound", ascending=False).head(40))


sns.histplot(data=df, x="wilson_lower_bound", bins=10)
print(plt.show())

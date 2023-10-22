import pandas as pd
import datetime as dt
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)


# Sorting Products: Ürün Sıralama:
df = pd.read_csv("product_sorting.csv")
print(df.head())

# Sorting by Rating:
df.sort_values("rating", ascending=False).head(10)

# Sorting by comment count and purchase count:
df.sort_values("purchase_count", ascending=False).head(10)
df.sort_values("commment_count", ascending=False).head(10)


# Sorting by rating, comment and purchase count:
df["comment_count_scaled"] = (MinMaxScaler(feature_range=(1, 5)).
                              fit(df[["commment_count"]]).
                              transform(df[["commment_count"]]))


df["purchase_count_scaled"] = (MinMaxScaler(feature_range=(1, 5)).
                               fit(df[["purchase_count"]]).
                               transform(df[["purchase_count"]]))


sorting_by_rpc = (df["rating"] * 42 / 100 +
                  df["purchase_count_scaled"] * 26 / 100 +
                  df["comment_count_scaled"] * 32 / 100)


def sorting_rpc(dataframe, w1=int(input("rating w")), w2=int(input("purchase w")), w3=int(input("comment w"))):
    if w1 + w2 + w3 != 100:
        return "Enter Weight Values Correctly!"
    else:
        return (dataframe["rating"] * w1 / 100 +
                dataframe["purchase_count_scaled"] * w2 / 100 +
                dataframe["comment_count_scaled"] * w3 / 100)

sorting_rpc(df)

df["sorting_by_rpc"] = sorting_rpc(df)
df.head()
df.sort_values("sorting_by_rpc", ascending=False).head(20)

# Bayesian Average Rating Score:
# Calculate a score according to distribution of points (from 1 to 5 points dist.)
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0

    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)

    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("sorting_by_rpc", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)

# Hybrid Sorting: BAR Score + Other Factors
def hybrid_sorting_score(dataframe, bar_w = int(input("bar weight")), rpc_w = int(input("rpc weight"))):
    if bar_w + rpc_w != 100:
        return "Enter weight values correctly!"
        pass
    else:
        bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)
        rpc_score = sorting_rpc(dataframe)
        return bar_score * bar_w / 100 + rpc_score * rpc_w / 100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(10)
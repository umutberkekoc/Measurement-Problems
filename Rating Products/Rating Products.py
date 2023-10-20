import pandas as pd
import datetime as dt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
df_ = pd.read_csv("course_reviews.csv")
df = df_.copy()
df.head()

# Data Understanding:
df.isnull().sum()
df["Rating"].value_counts() #frequency of rating
df["Questions Asked"].value_counts()  # frequency of questions asked

df.groupby("Questions Asked").agg({"Rating": "mean",
                                   "Questions Asked": "count"})

df.describe().T
df["Progress"].value_counts()
df = df[df["Progress"] > 0]   # Kursa Hiç Başlaamyan kişileri verisetinden çıkardık.
df.describe().T
# Rating Average
df["Rating"].mean()


# Time-Based Weighted Average:
df.columns
df.info()
df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")

df["Timestamp"].max()

today_date = dt.datetime(year=2021, month=2, day=7)

df["rating_day"] = (today_date - df["Timestamp"]).dt.days
df.head()

df["rating_day"].describe().T

w1 = 0
w2 = 0
w3 = 0
w4 = 0
for i in range(1):
    w1 = int(input("Enter W1 Value"))
    w2 = int(input("Enter W2 Value"))
    w3 = int(input("Enter W3 Value"))
    w4 = 100 - (w1 + w2 + w3)
    if w1 + w2 + w3 + w4 != 100:
        break

time_based_weighted_average = df[df["rating_day"] <= df["rating_day"].describe().T[4]]["Rating"].mean() * w1 / 100 + \
df[(df["rating_day"] > df["rating_day"].describe().T[4]) & (df["rating_day"] <= df["rating_day"].describe().T[5])]["Rating"].mean() * w2 / 100 + \
df[(df["rating_day"] > df["rating_day"].describe().T[5]) & (df["rating_day"] <= df["rating_day"].describe().T[6])]["Rating"].mean() * w3 / 100 + \
df[df["rating_day"] > df["rating_day"].describe().T[6]]["Rating"].mean() * w4 / 100



def time_based_weighted_average_calculator(dataframe, variable="rating_day", w1=int(input("w1")), w2=int(input("w2")), w3=int(input("w3")), w4=int(input("w4"))):
    if w1 + w2 + w3 + w4 != 100:
        return "Enter w Values Correctly!"
        pass
    else:
        return dataframe[dataframe[variable] <= dataframe[variable].describe().T[4]]["Rating"].mean() * w1 / 100 + \
dataframe[(dataframe[variable] > dataframe[variable].describe().T[4]) & (dataframe[variable] <= dataframe[variable].describe().T[5])]["Rating"].mean() * w2 / 100 + \
dataframe[(dataframe[variable] > dataframe[variable].describe().T[5]) & (dataframe[variable] <= dataframe[variable].describe().T[6])]["Rating"].mean() * w3 / 100 + \
dataframe[dataframe[variable] > dataframe[variable].describe().T[6]]["Rating"].mean() * w4 / 100

time_based_weighted_average_calculator(df)


# User-Based Weighted Average:
df.head()

df["Progress"].describe().T
w1 = 0
w2 = 0
w3 = 0
w4 = 0
for i in range(1):
    w1 = int(input("Enter W1 Value"))
    w2 = int(input("Enter W2 Value"))
    w3 = int(input("Enter W3 Value"))
    w4 = 100 - (w1 + w2 + w3)
    if w1 + w2 + w3 + w4 != 100:
        break

user_based_weighted_average = df[df["Progress"] <= df["Progress"].describe().T[4]]["Rating"].mean() * w1 / 100 + \
df[(df["Progress"] > df["Progress"].describe().T[4]) & (df["Progress"] <= df["Progress"].describe().T[5])]["Rating"].mean() * w2 / 100 + \
df[(df["Progress"] > df["Progress"].describe().T[5]) & (df["Progress"] <= df["Progress"].describe().T[6])]["Rating"].mean() * w3 / 100 + \
df[df["Progress"] > df["Progress"].describe().T[6]]["Rating"].mean() * w4 / 100

def user_based_weigted_average_calculator(dataframe, variable="Progress", w1 = int(input("w1")), w2 = int(input("w2")), w3 = int(input("w3")), w4=int(input("w4"))):
    if w1 + w2 + w3 + w4 != 100:
        return "Enter w Values Correctly!"
        pass
    else:
        return dataframe[dataframe[variable] <= dataframe[variable].describe().T[4]]["Rating"].mean() * w1 / 100 + \
            dataframe[(dataframe[variable] > dataframe[variable].describe().T[4]) & (dataframe[variable] <= dataframe[variable].describe().T[5])]["Rating"].mean() * w2 / 100 + \
            dataframe[(dataframe[variable] > dataframe[variable].describe().T[5]) & (dataframe[variable] <= dataframe[variable].describe().T[6])]["Rating"].mean() * w3 / 100 + \
            dataframe[dataframe[variable] > dataframe[variable].describe().T[6]]["Rating"].mean() * w4 / 100

user_based_weigted_average_calculator(df)

# Weighted Rating:
def weighted_rating_calculator(dataframe, time_base_weight = int(input("time based weight")), user_based_weight=int(input("user based weight"))):
    if time_base_weight + user_based_weight != 100:
        return "Enter Weight Values Correctly!"
        pass
    return time_based_weighted_average_calculator(dataframe) * time_base_weight / 100 + \
        user_based_weigted_average_calculator(dataframe) * user_based_weight / 100

weighted_rating_calculator(df)

df = pd.read_csv("movies_metadata.csv", low_memory="False") # DtypeWarning kapamak için
df.columns
df.head()
df.shape
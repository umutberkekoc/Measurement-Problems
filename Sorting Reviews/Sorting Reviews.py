# Sorting Reviews:

# Up-Down Difference Score: Üst-Alt Farkı Skoru ( up ratings - down ratings)
# Up --> Like, Down --> Dislike

# Example:
# Review 1: 600 up (like) 400 down (dislike) total 1000
# Review 2: 5500 up, 4500 down total 10000

import scipy.stats as st
import math
import pandas as pd

def up_down_diff_score(up, down):
    if up + down == 0:
        return 0
    else:
        return up -down

up_down_diff_score(600, 400)
up_down_diff_score(5500, 4500)
# According to results, review 1 is better than review 2

def average_up_score(up, down):
    if up + down == 0:
        return 0
    else:
        return up / (up + down)
average_up_score(600, 400)
average_up_score(up=5500, down=4500)
# According to results, review 1 is better than review 2

average_up_score(up=2, down=0)
average_up_score(up=100, down=1)
# According to results, review 1 is better than review 2
# But it seems does not make sense!!! We should consider frequency

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score Hesapla
    - Bernoulli parametresi 'p' için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasındaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve Bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------

    """
    n = up + down
    if n == 0:
        return 0
    else:
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        phat = up / n
        return (phat + z ** 2 / (2 * n) - z * math.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)) / (1 + z ** 2 / n)

wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)


# Mini Case:
up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1]

# up-down diff score
def up_down_diff_score(up, down):
    if up + down == 0:
        return 0
    else:
        return up - down


# average up score
def average_up_score(up, down):
    if up + down == 0:
        return 0
    else:
        return up / (up + down)

# wilson lower bound
def wilson_lower_bound(up, down, confidence = 0.95):
    n = up + down
    if n == 0:
        return 0
    else:
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        phat = up / n
        return (phat + z ** 2 / (2 * n) - z * math.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)) / (1 + z ** 2 / n)

df = pd.DataFrame({"up": up,
                   "down": down})

df["up_down_diff_score"] = df.apply(lambda x: up_down_diff_score(x["up"], x["down"]), axis=1)

df["average_up_score"] = df.apply(lambda  x: average_up_score(x["up"], x["down"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)

df.sort_values(by="wilson_lower_bound", ascending=False)

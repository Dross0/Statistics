import numpy as np
import scipy.stats as st
import math
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def getS1(data, a):
    """
    Вычисляет выборочную дисперсию, при существующем математическом ожидании
    :param data: Выборка
    :param a: Математическое ожидание
    :return: Выборочную дисперсию
    """
    res = 0.0
    for elem in data:
        res += (elem - a) ** 2
    return res / len(data)


def getAWithSigma(data, sigma2, e):
    """
    Вычисляет доверительный интервал для математического ожидания при известной дисперсии
    :param data: Выборка
    :param sigma2: Дисперсия
    :param e: epsilon
    :return: Доверительный интервал для математического ожидания
    """
    x = np.mean(data)
    quantile = st.norm.ppf(1 - (e / 2))
    print(f't(1-e/2) = {quantile}')
    n = len(data)
    sigma = sigma2 ** 0.5
    sqrN = n ** 0.5
    return x - (quantile * sigma / sqrN), x + (quantile * sigma / sqrN)


def getAWithoutSigma(data, e):
    """
    Вычисляет доверительный интервал для математического ожидания при неизвестной дисперсии
    :param data: Выборка
    :param e: epsilon
    :return: Доверительный интервал для математического ожидания
    """
    x = np.mean(data)
    n = len(data)
    S2 = np.var(data)
    S2_0 = n * S2 / (n - 1)
    S0 = S2_0 ** 0.5
    sqrN = n ** 0.5
    studentQuantile = st.t.ppf(1 - e / 2, n - 1)
    print(f'T(1 - e/2, n - 1) = {studentQuantile}')
    return x - (studentQuantile * S0 / sqrN), x + (studentQuantile * S0 / sqrN)


def getSigmaWithoutA(data, e):
    """
    Вычисляет доверительный интервал для дисперсии при неизвестном математическом ожидании
    :param data: Выборка
    :param e: epsilon
    :return: Доверительный интервал для дисперсии
    """
    S2 = np.var(data)
    n = len(data)
    XiQuantile1 = st.chi2.ppf(1 - e / 2, n - 1)
    XiQuantile2 = st.chi2.ppf(e / 2, n - 1)
    print(f'Xi(1 - e/2, n - 1) = {XiQuantile1}')
    print(f'Xi(e / 2, n - 1) = {XiQuantile2}')
    return n * S2 / XiQuantile1, n * S2 / XiQuantile2


def getSigmaWithA(data, a, e):
    """
    Вычисляет доверительный интервал для дисперсии при известном математическом ожидании
    :param data: Выборка
    :param a: Математическое ожидание
    :param e: epsilon
    :return: Доверительный интервал для дисперсии
    """
    S2_1 = getS1(data, a)
    n = len(data)
    XiQuantile1 = st.chi2.ppf(1 - e / 2, n)
    XiQuantile2 = st.chi2.ppf(e / 2, n)
    print(f'Xi(1 - e/2, n) = {XiQuantile1}')
    print(f'Xi(e / 2, n) = {XiQuantile2}')
    return n * S2_1 / XiQuantile1, n * S2_1 / XiQuantile2


def DICount(data, a, sigma2, e):
    n = len(data)
    print(f"N = {n}")
    print(f"X = {np.mean(data)}")
    print(f"S^2 = {np.var(data)}")
    print(f"S0^2 = {np.var(data) * n / (n - 1)}")
    print(f"S1^2 = {getS1(data, a)}")
    DIforAWithSigma = getAWithSigma(data, sigma2, e)
    print("ДИ для a если известна sigma^2: " + str(DIforAWithSigma))
    DIforAWithoutSigma = getAWithoutSigma(data, e)
    print("ДИ для a если sigma^2 неизвестна: " + str(DIforAWithoutSigma))
    DIforSigmaWithoutA = getSigmaWithoutA(data, e)
    print("ДИ для sigma^2 если а неизвестно: " + str(DIforSigmaWithoutA))
    DIforSigmaWithA = getSigmaWithA(data, a, e)
    print("ДИ для sigma^2 если а известно: " + str(DIforSigmaWithA))


def fisherCriterion(data1, data2, e):
    """
    Критерий Фишера - проверяет гипотезу о равенстве двух выборок
    :param data1: Выборка X
    :param data2: Выборка Y
    :param e: epsilon
    :return: True, если верна основная гипотеза (о равенстве дисперсий двух выборок), False иначе
    """
    n = len(data1)
    m = len(data2)
    S0X = np.var(data1) * n / (n - 1)
    S0Y = np.var(data2) * m / (m - 1)
    print(f'S_0(X) = {S0X}')
    print(f'S_0(Y) = {S0Y}')
    d = S0X / S0Y if S0X >= S0Y else S0Y / S0X
    f1 = st.f.ppf(e / 2, n - 1, m - 1)
    f2 = st.f.ppf(1 - e / 2, n - 1, m - 1)
    print(f'f(e/2) = {f1}')
    print(f'f(1 - e/2) = {f2}')
    print(f'd = {d}')
    return f1 < d < f2


def studentCriterion(data1, data2, e):
    """
    Критерий Стьюдента - проверяет гипотезу о равенстве математических ожиданий двух выборок
    :param data1: Выборка X
    :param data2: Выборка Y
    :param e: epsilon
    :return: True, если верна основная гипотеза (о равенстве математических ожиданий двух выборок), False иначе
    """
    n = len(data1)
    m = len(data2)
    SX = np.var(data1)
    SY = np.var(data2)
    avgX = np.mean(data1)
    avgY = np.mean(data2)
    print(f'X = {avgX}')
    print(f'Y = {avgY}')
    numerator = (avgX - avgY) * ((m + n - 2) ** 0.5) * ((n * m) ** 0.5)
    denominator = ((n + m) ** 0.5) * ((n * SX + m * SY) ** 0.5)
    d = numerator / denominator
    # d = st.ttest_ind(data1, data2)
    t = st.t.ppf(1 - e / 2, n + m - 2)
    print(f't(1 - e /2) = {t}')
    print(f'd = {d}')
    return abs(d) < t


def crit(data1, data2, e):
    fCrit = fisherCriterion(data1, data2, e)
    print("По критерию Фишера дисперсии " + ('' if fCrit else 'не ') + "совпадают")
    sCrit = studentCriterion(data1, data2, e)
    print("По критерию Стьюдента выборочные средние " + ('' if sCrit else 'не ') + "совпадают")


def KolmagorovCrit(data, e, distribution):
    """
    Критерий Колмогорова - проверяет гипотезу, о принадлежности случайной величины распределению (distribution)
    :param distribution: Распределение
    :param data: Выборка
    :param e: epsilon
    :return: True, если верна основная гипотеза (о принадлежности случайной величины распределению), False иначе
    """
    stat, pvalue = st.kstest(data, distribution)
    d = (len(data) ** 0.5) * stat
    print(f"Статистика критерия Колмагорова (d) =  {d}")
    print(f'РДУЗ = {pvalue}')
    c = st.kstwobign.ppf(1 - e)
    print(f'c = {c}')
    return d < c


def PirsonCrit(data, e):
    """
    Критерий Пирсона (Хи - квадрат) - проверяет гипотезу, о принадлежности случайной величины
     теоритическому закону распределения
    :param data: Выборка
    :param e: epsilon
    :return: True, если верна основная гипотеза (о принадлежности случайной величины теоритическому закону распределения),
    False иначе
    """
    hist, bins = np.histogram(data, bins=math.floor(math.log2(len(data)) + 1))
    print(hist, bins)
    k = len(hist)
    n = len(data)
    d = 0.0
    for i in range(k):
        nP = n * (bins[i + 1] - bins[i])
        d += ((hist[i] - nP) ** 2) / nP
    print(f'Статистика критерия Пирсона (d) = {d}')
    x = 1 - st.chi2.cdf(d, k - 1)
    print(f'РДУЗ = {x}')
    c = st.chi2.ppf(1 - e, k - 1)
    print(f'c = {c}')
    return d < c


def critForUniform(data, e):
    ktest = KolmagorovCrit(data, e, 'uniform')
    ptest = PirsonCrit(data, e)
    print(f'Критерий Колмагорова ' + ('подтверждает' if ktest else 'отвергает') + ' основную гипотезу')
    print(f'Критерий Пирсона ' + ('подтверждает' if ptest else 'отвергает') + ' основную гипотезу')


def getSupForUniform(data):
    """
    Вычисляет супремум между эмперической и теоритической функцией распределения
    :param data: Выборка из равномерного распределения
    :return: sup - супремум между эмперической и теоритической функцией распределения, point - точка,
     в которой достигается супремум
    """
    ecdf = ECDF(data)
    sup = float('-Inf')
    point = 0
    for i in range(len(data)):
        pointSub = math.fabs(ecdf.y[i + 1] - st.uniform.cdf(ecdf.x[i + 1]))
        if sup < pointSub:
            sup = pointSub
            point = ecdf.x[i + 1]
    return sup, point

def showHistAndECDF(data):
    '''
    Рисует график эмпирической функции распределения и гистограмму
    :param data: Выборка
    :return: None
    '''
    ecdf = ECDF(data)
    plt.step(ecdf.x, ecdf.y)
    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y)
    plt.show()
    plt.hist(data, bins=math.floor(math.log2(len(data)) + 1))
    plt.show()

def main():
    data = list(map(float, input().split()))
    a = float(input('Математическое ожидание: '))
    sigma2 = float(input('Дисперсия: '))
    e = float(input('Epsilon = '))
    DICount(data, a, sigma2, e)
    crit(data[:20], data[20:], e)
    uniformData = list(map(float, input().split()))
    critForUniform(uniformData, e)
    showHistAndECDF(uniformData)
    sup, point = getSupForUniform(uniformData)
    print(f'Супремум = {sup} в точке ({point})')


if __name__ == '__main__':
    main()

import numpy as np

eps = 0.001
iteration = 0

def cubic(x):
    return x**3


def absolute(x):
    return np.abs(x-0.2)


def sinus(x):
    return x*np.sin(1/x)


def brute_force(func, a, b):
    iteration = 0
    n = int(np.floor((b-a)/eps))
    minimum = []
    for i in range(n+1):
        iteration += 1
        minimum.append(func(a+i*(b-a)/n))
    val, idx = min((val, idx) for (idx, val) in enumerate(minimum))
    return a + idx * (b - a) / n, iteration


def dichotomy(func, a, b, iteration):
    if np.abs(a - b) < eps:
        return a, b, iteration
    else:
        x1 = (a+b-eps/2)/2
        x2 = (a+b+eps/2)/2
        if func(x1) <= func(x2):
            b = x2
        else:
            a = x1
        iteration += 1
        return dichotomy(func, a, b, iteration)


def golden_ratio_first(func, a, b):
    if np.abs(a - b) < eps:
        return a, b, 0
    else:
        x1 = a + ((3 - np.sqrt(5)) / 2) * (b - a)
        x2 = b + ((np.sqrt(5) - 3) / 2) * (b - a)
        if func(x1) <= func(x2):
            b = x2
            return golden_ratio(func, a, b, x1, True, 1)
        else:
            a = x1
            return golden_ratio(func, a, b, x2, False, 1)


def golden_ratio(func, a, b, calcX, check, iteration):
    if np.abs(a - b) < eps:
        return a, b, iteration
    else:
        iteration += 1
        if check:
            x1 = a + ((3 - np.sqrt(5)) / 2) * (b - a)
            x2 = calcX
        else:
            x1 = calcX
            x2 = b + ((np.sqrt(5) - 3) / 2) * (b - a)
        if func(x1) <= func(x2):
            b = x2
            return golden_ratio(func, a, b, x1, True, iteration)
        else:
            a = x1
            return golden_ratio(func, a, b, x2, False, iteration)


print(brute_force(cubic, 0, 1))
print(brute_force(absolute, 0, 1))
print(brute_force(sinus, 0.01, 1))
print(dichotomy(cubic, 0, 1, 0))
print(dichotomy(absolute, 0, 1, 0))
print(dichotomy(sinus, 0.01, 1, 0))
print(golden_ratio_first(cubic, 0, 1))
print(golden_ratio_first(absolute, 0, 1))
print(golden_ratio_first(sinus, 0.01, 1))

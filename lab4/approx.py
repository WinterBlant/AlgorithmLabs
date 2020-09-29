import numpy as np
from scipy.optimize import minimize, least_squares, differential_evolution, dual_annealing
from matplotlib import pyplot as plt


def rational(a, b, c, d, x):
    return (a * x + b) / (x ** 2 + c * x + d)


def lst_sq_func(x, a, b):
    result = 0
    for i in range(1001):
        result += (((x[0] * a[i] + x[1]) / (a[i] ** 2 + x[2] * a[i] + x[3])) - b[i]) ** 2
    return result


def lst_sq_func_lm(x, a, b):
    result = []
    for i in range(1001):
        result.append(((x[0] * a[i] + x[1]) / (a[i] ** 2 + x[2] * a[i] + x[3])) - b[i])
    return result


def func(x):
    return 1 / (x ** 2 - 3 * x + 2)


delta = np.random.normal(0, 1, 1001)
x0 = np.array([1, 1, 1, 1])
y = []
x = []
y_nelder = []
y_lm = []
y_diff = []
y_anneal = []
bounds = [(-2,2), (-2,2), (-2,2), (-2,2)]
for k in range(1001):
    t = func(3 * k / 1000)
    if t < -100:
        y.append(-100 + delta[k])
    elif t > 100:
        y.append(100 + delta[k])
    else:
        y.append(t + delta[k])
    x.append(3 * k / 1000)
result_nelder = minimize(lst_sq_func, x0=x0, args=(x, y), method="Nelder-Mead", options={'disp':True, 'fatol':0.001, 'maxiter':1000}).x
result_lm = least_squares(lst_sq_func_lm, x0=x0, args=(x, y), method="lm", max_nfev=1000, ftol=None, xtol=0.001, gtol=None, verbose=1)
print(lst_sq_func(result_lm.x, x, y))
result_diff = differential_evolution(lst_sq_func, bounds=bounds, args=(x, y), maxiter=1000, tol=0.001)
print(result_diff.nit, result_diff.nfev, lst_sq_func(result_diff.x, x, y))
result_anneal = dual_annealing(lst_sq_func, bounds=bounds, args=(x, y), maxiter=1000)
print(result_anneal.nit, result_anneal.nfev, lst_sq_func(result_anneal.x, x, y))
for k in range(1001):
    y_nelder.append(rational(result_nelder[0], result_nelder[1], result_nelder[2], result_nelder[3], x[k]))
    y_lm.append(rational(result_lm.x[0], result_lm.x[1], result_lm.x[2], result_lm.x[3], x[k]))
    y_diff.append(rational(result_diff.x[0], result_diff.x[1], result_diff.x[2], result_diff.x[3], x[k]))
    y_anneal.append(rational(result_anneal.x[0], result_anneal.x[1], result_anneal.x[2], result_anneal.x[3], x[k]))
plt.figure(figsize=[10, 6])
plt.scatter(x, y, label="Сгенерированные данные")
plt.plot(x, y_nelder, label="Метод Нелдера-Мида", color="red")
plt.plot(x, y_lm, label="Метод Левенберга-Марквардта", color="orange")
plt.plot(x, y_diff, label="Дифференциальная эволюция", color="pink")
plt.plot(x, y_anneal, label="Имитация отжига", color="green")
plt.plot()
plt.legend()
plt.savefig("approx.png", dpi=1000, quality=95)
plt.show()
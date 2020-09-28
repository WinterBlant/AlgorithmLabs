import numpy as np
from scipy.optimize import minimize, brute
from matplotlib import pyplot as plt


def linear(a, b, x):
    return a * x + b


def rational(a, b, x):
    return a / (1 + b * x)


def lst_sq_linear_func(x, a, b):
    result = 0
    for i in range(101):
        result += (a[i]*x[0]+x[1] - b[i])**2
    return result


def lst_sq_rational_func(x, a, b):
    result = 0
    for i in range(101):
        result += (x[0]/(1+x[1]*a[i]) - b[i])**2
    return result


alpha = np.random.random()
beta = np.random.random()
delta = np.random.normal(0, 1, 101)
x0 = np.array([1, 1])
y = []
x = []
y_alpha = []
y_brute = []
y_powell = []
y_nelder = []
y_brute_rational = []
y_powell_rational = []
y_nelder_rational = []
for k in range(101):
    y.append(alpha*(k/100) + beta + delta[k])
    y_alpha.append(alpha*(k/100) + beta)
    x.append(k/100)
ranges = (slice(0, 1, 0.01), slice(min(y), max(y), 0.01))
result_brute = brute(lst_sq_linear_func, ranges=ranges, args=(x, y), disp=True, full_output=True)
result_powell = minimize(lst_sq_linear_func, x0=x0, args=(x, y), method="Powell", options={'disp':True, 'ftol':0.001}).x
result_nelder = minimize(lst_sq_linear_func, x0=x0, args=(x, y), method="Nelder-Mead", options={'disp':True, 'fatol':0.001}).x
result_brute_rational = brute(lst_sq_rational_func, ranges=ranges, args=(x, y), disp=True, full_output=True)
result_powell_rational = minimize(lst_sq_rational_func, x0=x0, args=(x, y), method="Powell", options={'disp':True, 'ftol':0.001}).x
result_nelder_rational = minimize(lst_sq_rational_func, x0=x0, args=(x, y), method="Nelder-Mead", options={'disp':True, 'fatol':0.001}).x
for k in range(101):
    y_brute.append(linear(result_brute[0][0], result_brute[0][1], x[k]))
    y_powell.append(linear(result_powell[0], result_powell[1], x[k]))
    y_nelder.append(linear(result_nelder[0], result_nelder[1], x[k]))
    y_brute_rational.append(rational(result_brute_rational[0][0], result_brute_rational[0][1], x[k]))
    y_powell_rational.append(rational(result_powell_rational[0], result_powell_rational[1], x[k]))
    y_nelder_rational.append(rational(result_nelder_rational[0], result_nelder_rational[1], x[k]))
plt.figure(figsize=[10, 6])
plt.scatter(x, y, label="Сгенерированные данные")
plt.plot(x, y_alpha, label="Порождающая прямая", color="red")
plt.plot(x, y_brute, label="Метод перебора", color="orange")
plt.plot(x, y_powell, label="Метод Пауэлла", color="pink")
plt.plot(x, y_nelder, label="Метод Нелдера-Мида", color="green")
plt.plot()
plt.legend()
plt.savefig("linear.png", dpi=1000, quality=95)
plt.show()
plt.clf()
plt.figure(figsize=[10, 6])
plt.scatter(x, y, label="Сгенерированные данные")
plt.plot(x, y_alpha, label="Порождающая прямая", color="red")
plt.plot(x, y_brute_rational, label="Метод перебора", color="orange")
plt.plot(x, y_powell_rational, label="Метод Пауэлла", color="pink")
plt.plot(x, y_nelder_rational, label="Метод Нелдера-Мида", color="green")
plt.plot()
plt.legend()
plt.savefig("rational.png", dpi=1000, quality=95)
plt.show()

with open("params.txt", "a") as f:
    f.write(str(alpha)+" "+str(beta)+'\n')
    f.write(str(delta))

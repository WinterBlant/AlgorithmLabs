import numpy as np
from scipy.optimize import minimize, least_squares, minimize_scalar
from matplotlib import pyplot as plt


def jacobian_linear(x, a, b):
    result_dx, result_dy = 0, 0
    for i in range(101):
        result_dx += 2 * a[i] * (a[i] * x[0] + x[1] - b[i])
        result_dy += 2 * (a[i] * x[0] + x[1] - b[i])
    return np.array((result_dx, result_dy))


def jacobian_rational(x, a, b):
    result_dx, result_dy = 0, 0
    for i in range(101):
        result_dx -= 2 * (a[i] * b[i] * x[1] + b[i] - x[0]) / (a[i] * x[1] + 1) ** 2
        result_dy += 2 * a[i] * x[0] * (a[i] * b[i] * x[1] + b[i] - x[0]) / (a[i] * x[1] + 1) ** 3
    return np.array((result_dx, result_dy))


def hessian_linear(x, a, b):
    result_dx2, result_dxdy, result_dydx, result_dy2 = 0, 0, 0, 0
    for i in range(101):
        result_dx2 += 2 / (a[i] * x[1] + 1) ** 2
        result_dxdy += 2 * a[i] * (a[i] * b[i] * x[1] + b[i] - 2 * x[0]) / (a[i] * x[1] + 1) ** 3
        result_dydx += 2 * a[i] * (a[i] * b[i] * x[1] + b[i] - 2 * x[0]) / (a[i] * x[1] + 1) ** 3
        result_dy2 -= 2 * (a[i] ** 2) * x[0] * (2 * b[i] * (a[i] * x[1] + 1) - 3 * x[0]) / (a[i] * x[1] + 1) ** 4
    return np.array(((result_dx2, result_dxdy), (result_dydx, result_dy2)))


def hessian_rational(x, a, b):
    result_dx2, result_dxdy, result_dydx, result_dy2 = 0, 0, 0, 0
    for i in range(101):
        result_dx2 += 2 * a[i] ** 2
        result_dxdy += 2 * a[i]
        result_dydx += 2 * a[i]
        result_dy2 += 2
    return np.array(((result_dx2, result_dxdy), (result_dydx, result_dy2)))


def minimize_func_grad_linear(step, x, a, b, primes):
    result = 0
    for i in range(101):
        result += (a[i] * (x[0] - step * primes[0]) + (x[1] - step * primes[1]) - b[i]) ** 2
    return result


def minimize_func_grad_rational(step, x, a, b, primes):
    result = 0
    for i in range(101):
        result += ((x[0] - step * primes[0]) / (1 + (x[1] - step * primes[1]) * a[i]) - b[i]) ** 2
    return result


def gradient_descent(func, minimize_func, x_prev, jac, args=()):
    f_prime = jac(x_prev, *args)
    step = minimize_scalar(minimize_func, method="brent", args=(x_prev, *args, f_prime))
    print(step.nit, step.nfev)
    x_new = np.array((x_prev[0] - step.x * f_prime[0], x_prev[1] - step.x * f_prime[1]))
    iteration = 1
    while np.abs(func(x_new, *args) - func(x_prev, *args)) > 0.001:
        x_prev = x_new
        f_prime = jac(x_prev, *args)
        step = minimize_scalar(minimize_func, method="brent", args=(x_prev, *args, f_prime))
        print(step.nit, step.nfev)
        x_new = np.array((x_prev[0] - step.x * f_prime[0], x_prev[1] - step.x * f_prime[1]))
        iteration += 1
    return x_new, iteration


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


def lst_sq_linear_func_lm(x, a, b):
    result = []
    for i in range(101):
        result.append(a[i] * x[0] + x[1] - b[i])
    return np.array(result)


def lst_sq_rational_func_lm(x, a, b):
    result = []
    for i in range(101):
        result.append((x[0] / (1 + x[1] * a[i])) - b[i])
    return np.array(result)


delta = []
with open("params.txt", "r") as f:
    param = f.readline().split(" ")
    alpha = float(param[0])
    beta = float(param[1])
    for elem in f.readline()[1:-2].split(" "):
        delta.append(float(elem))
x0 = np.array([1, 1])
y = []
x = []
y_alpha = []
y_gd = []
y_cg = []
y_newton = []
y_lm = []
y_gd_rational = []
y_cg_rational = []
y_newton_rational = []
y_lm_rational = []
for k in range(101):
    y.append(alpha*(k/100) + beta + delta[k])
    y_alpha.append(alpha*(k/100) + beta)
    x.append(k/100)
result_gd, gd_iter = gradient_descent(lst_sq_linear_func, minimize_func_grad_linear, x_prev=x0, jac=jacobian_linear, args=(x, y))
print(gd_iter, lst_sq_linear_func(result_gd, x, y))
result_cg = minimize(lst_sq_linear_func, x0=x0, args=(x, y), method="CG", jac=jacobian_linear, options={'disp':True, 'gtol':0.001}).x
result_newton = minimize(lst_sq_linear_func, x0=x0, args=(x, y), method="Newton-CG", jac=jacobian_linear, hess=hessian_linear, options={'disp': True, 'xtol': 0.001}).x
result_lm = least_squares(lst_sq_linear_func_lm, x0=x0, args=(x, y), method="lm", ftol=None, xtol=0.001, gtol=None, verbose=1)
print(lst_sq_linear_func(result_lm.x, x, y))
result_gd_rational, gd_iter_rational = gradient_descent(lst_sq_rational_func, minimize_func_grad_rational, x_prev=x0, jac=jacobian_rational, args=(x, y))
print(gd_iter_rational, lst_sq_rational_func(result_gd_rational, x, y))
result_cg_rational = minimize(lst_sq_rational_func, x0=x0, args=(x, y), method="CG", jac=jacobian_rational, options={'disp':True, 'gtol':0.001}).x
result_newton_rational = minimize(lst_sq_rational_func,  x0=x0, args=(x, y), method="Newton-CG", jac=jacobian_rational, hess=hessian_rational, options={'disp':True, 'xtol':0.001}).x
result_lm_rational = least_squares(lst_sq_rational_func_lm, x0=x0, args=(x, y), method="lm", ftol=None, xtol=0.001, gtol=None, verbose=1)
print(lst_sq_rational_func(result_lm_rational.x, x, y))
for k in range(101):
    y_gd.append(linear(result_gd[0], result_gd[1], x[k]))
    y_cg.append(linear(result_cg[0], result_cg[1], x[k]))
    y_newton.append(linear(result_newton[0], result_newton[1], x[k]))
    y_lm.append(linear(result_lm.x[0], result_lm.x[1], x[k]))
    y_gd_rational.append(rational(result_gd_rational[0], result_gd_rational[1], x[k]))
    y_cg_rational.append(rational(result_cg_rational[0], result_cg_rational[1], x[k]))
    y_newton_rational.append(rational(result_newton_rational[0], result_newton_rational[1], x[k]))
    y_lm_rational.append(rational(result_lm_rational.x[0], result_lm_rational.x[1], x[k]))
plt.figure(figsize=[10, 6])
plt.scatter(x, y, label="Сгенерированные данные")
plt.plot(x, y_alpha, label="Порождающая прямая", color="red")
plt.plot(x, y_gd, label="Градиентный спуск", color="orange")
plt.plot(x, y_cg, label="Метод сопряженных градиентов", color="pink")
plt.plot(x, y_newton, label="Метод Ньютона", color="green")
plt.plot(x, y_lm, label="Метод Левенберга-Марквадта", color="black")
plt.plot()
plt.legend()
plt.savefig("linear.png", dpi=1000, quality=95)
plt.show()
plt.clf()
plt.figure(figsize=[10, 6])
plt.scatter(x, y, label="Сгенерированные данные")
plt.plot(x, y_alpha, label="Порождающая прямая", color="red")
plt.plot(x, y_gd_rational, label="Градиентный спуск", color="orange")
plt.plot(x, y_cg_rational, label="Метод сопряженных градиентов", color="pink")
plt.plot(x, y_newton_rational, label="Метод Ньютона", color="green")
plt.plot(x, y_lm_rational, label="Метод Левенберга-Марквадта", color="black")
plt.plot()
plt.legend()
plt.savefig("rational.png", dpi=1000, quality=95)
plt.show()
